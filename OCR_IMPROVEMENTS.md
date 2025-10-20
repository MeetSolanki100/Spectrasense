# OCR Improvements - Enhanced Text Extraction

## Overview
The vision encoder now uses **Pytesseract** with advanced preprocessing techniques to dramatically improve text extraction accuracy for both **printed** and **handwritten** text.

## Key Improvements

### 1. **Image Preprocessing Pipeline**
- **Grayscale Conversion**: Simplifies image for better text detection
- **Adaptive Thresholding**: Improves text contrast against varying backgrounds
- **Denoising**: Removes image noise that can interfere with OCR
- **Dilation**: Makes text characters more readable

### 2. **Multi-Mode OCR Strategy**
The system now tries **4 different OCR approaches** and returns the best result:

#### Mode 1: Original Image (PSM 3)
- **Config**: `--psm 3 --oem 3`
- **Best for**: Standard documents with mixed text layouts
- **PSM 3**: Fully automatic page segmentation

#### Mode 2: Preprocessed Image (PSM 3)
- **Config**: `--psm 3 --oem 3` on preprocessed image
- **Best for**: Printed text with clear backgrounds
- **Advantage**: Enhanced contrast improves accuracy

#### Mode 3: Single Block Text (PSM 6)
- **Config**: `--psm 6 --oem 3`
- **Best for**: Uniform blocks of text (paragraphs, signs)
- **PSM 6**: Assumes single uniform block of text

#### Mode 4: Sparse Text (PSM 11)
- **Config**: `--psm 11 --oem 3`
- **Best for**: Scattered text, labels, signs
- **PSM 11**: Finds text in any order without structure assumptions

### 3. **Enhanced Text Detection with Bounding Boxes**
- Runs OCR on both original and preprocessed images
- Combines results to catch text missed by single approach
- Removes duplicates using position-based deduplication
- Sorts text blocks top-to-bottom, left-to-right for natural reading order
- Lower confidence threshold (0.3) to catch more text

### 4. **Text Cleaning**
- Removes empty lines
- Strips whitespace
- Returns "No text detected" instead of empty string for clarity

## Configuration Parameters

```python
TEXT_DETECTION_CONFIDENCE = 0.3  # Lowered from 0.5 to catch more text
```

### Tesseract PSM Modes Explained
- **PSM 0**: Orientation and script detection only
- **PSM 3**: Fully automatic page segmentation (default)
- **PSM 4**: Single column of text
- **PSM 6**: Single uniform block of text
- **PSM 7**: Single text line
- **PSM 8**: Single word
- **PSM 11**: Sparse text (find as much as possible)
- **PSM 13**: Raw line (bypass all segmentation)

### Tesseract OEM Modes
- **OEM 3**: Default, based on what is available (LSTM + Legacy)

## Usage

### API Response Format
```json
{
  "status": "success",
  "full_text": "Complete extracted text from image\nWith line breaks preserved",
  "text_blocks": [
    {
      "text": "Individual word or phrase",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y1, x2, y2, x1, y2]
    }
  ],
  "room_description": "AI-generated scene description",
  "objects": ["person", "laptop", "book"]
}
```

## Best Practices for Better OCR Results

### Image Quality
1. **Good Lighting**: Ensure text is well-lit and clearly visible
2. **Focus**: Keep camera steady and text in focus
3. **Angle**: Hold camera perpendicular to text (avoid skew)
4. **Distance**: Get close enough for text to be readable
5. **Resolution**: Higher resolution = better OCR accuracy

### Text Types Supported
- ✅ Printed text (books, documents, signs)
- ✅ Computer-generated text (screens, displays)
- ✅ Handwritten text (clear handwriting)
- ✅ Mixed layouts (forms, receipts)
- ✅ Multi-column text
- ✅ Sparse text (labels, tags)

### Known Limitations
- ⚠️ Very small text may be difficult to read
- ⚠️ Heavily stylized fonts may reduce accuracy
- ⚠️ Extremely poor handwriting may not be recognized
- ⚠️ Text on curved surfaces may be challenging
- ⚠️ Very low contrast text may be missed

## Testing the Improvements

### Real-Time Camera Feed
1. Navigate to http://localhost:5005/real_time
2. Click "Start Camera"
3. Point camera at text
4. View extracted text in the analysis panel

### Upload Image
1. Navigate to http://localhost:5005/upload_page
2. Upload an image with text
3. View full extracted text and bounding boxes

## Technical Details

### Dependencies
- **pytesseract**: Python wrapper for Tesseract OCR
- **tesseract**: OCR engine (installed via Homebrew)
- **opencv-python**: Image preprocessing
- **PIL/Pillow**: Image handling

### Performance
- Multiple OCR passes may take 2-3 seconds per frame
- Real-time mode processes every 3rd frame to maintain performance
- Preprocessing adds ~200ms per image

## Troubleshooting

### "No text detected"
- Ensure text is clearly visible and in focus
- Try adjusting camera angle and distance
- Check lighting conditions

### Low Accuracy
- Improve image quality (lighting, focus, angle)
- For handwritten text, ensure writing is clear
- Try uploading a higher resolution image

### Slow Performance
- Real-time mode automatically throttles processing
- For faster results, reduce image resolution
- Consider processing static images instead of video

## Future Enhancements
- [ ] Add language support beyond English
- [ ] Implement text orientation correction
- [ ] Add perspective correction for skewed text
- [ ] Support for vertical text (Asian languages)
- [ ] GPU acceleration for preprocessing
