import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Create a test directory if it doesn't exist
os.makedirs('test_images', exist_ok=True)

def create_test_image():
    # Create a white image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Add horizontal text
    try:
        font = ImageFont.truetype("Arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
    
    # Horizontal text
    draw.text((50, 100), "This is horizontal text", font=font, fill="black")
    draw.text((50, 150), "Another line of horizontal text", font=font, fill="black")
    
    # Vertical text (manually rotated)
    vertical_text = "VERTICAL"
    text_image = Image.new('RGBA', (40, 200), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((0, 0), vertical_text, font=font, fill="black")
    rotated_text = text_image.rotate(90, expand=1)
    image.paste(rotated_text, (500, 100), rotated_text)
    
    # Save the image
    test_image_path = 'test_images/test_ocr_image.jpg'
    image.save(test_image_path)
    return test_image_path

def test_ocr():
    from main import analyze_frame, initialize_models
    
    # Initialize models
    print("Initializing models...")
    yolo_model, blip_processor, blip_model, qwen_model, qwen_processor, easyocr_reader, device = initialize_models()
    
    # Create test image
    print("Creating test image...")
    test_image_path = create_test_image()
    print(f"Test image saved to: {test_image_path}")
    
    # Load and analyze the image
    print("Analyzing image...")
    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"Error: Could not load test image from {test_image_path}")
        return
    
    # Run the analysis
    result = analyze_frame(frame)
    
    # Print results
    print("\n--- Analysis Results ---")
    print(f"Caption: {result.get('caption', 'N/A')}")
    
    print("\nDetected Objects:")
    for obj in result.get('objects', []):
        if isinstance(obj, dict):
            print(f"- {obj.get('class', 'Unknown')} (Confidence: {obj.get('confidence', 0):.2f})")
        else:
            print(f"- {obj}")
    
    print("\nDetected Text:")
    for i, text_block in enumerate(result.get('text_blocks', []), 1):
        print(f"{i}. '{text_block.get('text', '')}'")
        print(f"   Orientation: {text_block.get('orientation', 'unknown')}")
        print(f"   Confidence: {text_block.get('confidence', 0):.2f}")
    
    print("\nHorizontal Text:")
    for i, text in enumerate(result.get('horizontal_text', []), 1):
        print(f"{i}. '{text}'")
    
    print("\nVertical Text:")
    for i, text in enumerate(result.get('vertical_text', []), 1):
        print(f"{i}. '{text}'")
    
    if 'error' in result and result['error']:
        print(f"\nError: {result['error']}")

if __name__ == "__main__":
    test_ocr()
