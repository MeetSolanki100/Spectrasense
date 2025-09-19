
from flask import Flask, render_template, Response
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np # Import numpy

app = Flask(__name__)

# Global variable to control monitoring state
monitoring_active = False
ANTI_SPOOF_THRESHOLD = 0.8 # Adjusted threshold for anti-spoofing (increased for stricter classification)

# Load the anti-spoofing model
model_path = '/Users/kabirmathur/Documents/a_s/antispoof_vit.pth'

class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=192):
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, num_classes=2,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=nn.LayerNorm)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.fc_norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x

anti_spoof_model = VisionTransformer(num_classes=2)
anti_spoof_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
anti_spoof_model.eval()

# Define preprocessing for the anti-spoofing model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # Reverted to 0.5 mean/std
])

# Initialize YOLO face detector
yolo_model = YOLO('yolov8n.pt') # Reverted to generic YOLOv8n model for stability

# Get class names from the YOLO model
class_names = yolo_model.names
face_class_id = None
for class_id, name in class_names.items():
    if name == 'person': # YOLOv8n typically detects 'person' for faces
        face_class_id = class_id
        break

if face_class_id is None:
    print("Warning: 'person' class not found in YOLO model. Face detection might not work as expected.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_monitoring')
def start_monitoring():
    global monitoring_active
    monitoring_active = True
    print("Monitoring started via Flask route.")
    return "Monitoring started"

@app.route('/stop_monitoring')
def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    print("Monitoring stopped via Flask route.")
    return "Monitoring stopped"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global monitoring_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream. Please check webcam permissions or if another application is using it.")
        # Generate a blank frame with an error message
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "Webcam Error: Check permissions or if in use."
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        while True:
            yield (b'--frame\n'
                   b'Content-Type: image/jpeg\n\n' + frame + b'\n')

    while True:
        if not monitoring_active:
            # If monitoring is not active, yield a blank frame or a static image
            # For now, we'll yield a blank gray frame
            blank_frame = np.full((480, 640, 3), 128, dtype=np.uint8) # Gray frame
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\n'
                   b'Content-Type: image/jpeg\n\n' + frame + b'\n')
            continue # Skip processing if not active

        success, frame = cap.read()
        if not success:
            break

        results = yolo_model(frame)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Filter for 'person' class if identified
                if face_class_id is not None and int(box.cls[0]) != face_class_id:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                input_tensor = preprocess(face_pil).unsqueeze(0)

                with torch.no_grad():
                    output = anti_spoof_model(input_tensor)
                    # Corrected: Assuming output is [logit_real, logit_fake] and 'real' is index 0.
                    probability = torch.sigmoid(output[:, 0]).item() # Take the first logit (index 0) and apply sigmoid
                    is_real = probability > ANTI_SPOOF_THRESHOLD # Use configurable threshold

                print(f"Face detected: Probability = {probability:.4f}, Is Real = {is_real}") # Debugging output

                # Only draw a green rectangle if the face is classified as real
                if is_real:
                    color = (0, 255, 0) # Green for real faces
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\n'
               b'Content-Type: image/jpeg\n\n' + frame + b'\n')

    cap.release()

if __name__ == "__main__":
    app.run(debug=True)
