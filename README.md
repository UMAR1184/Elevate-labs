# Day 7: Image Resizer Tool

A Python script that batch resizes and converts images in a folder using Pillow (PIL).

## Features

- **Batch Processing**: Resize all images in a folder at once
- **Multiple Formats**: Supports JPG, PNG, BMP, TIFF, WebP
- **Aspect Ratio**: Maintains aspect ratio by default (optional)
- **Format Conversion**: Convert between different image formats
- **Quality Control**: Adjustable JPEG quality settings
- **Smart Output**: Creates organized output folder structure

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python image_resizer.py /path/to/images
```

### Advanced Options
```bash
# Resize to specific dimensions
python image_resizer.py /path/to/images -w 1920 -h 1080

# Convert to different format
python image_resizer.py /path/to/images -f jpg -q 90

# Custom output folder
python image_resizer.py /path/to/images -o /path/to/output

# Don't maintain aspect ratio
python image_resizer.py /path/to/images --no-aspect
```

### Command Line Arguments

- `input_folder`: Path to folder containing images (required)
- `-o, --output`: Output folder (default: input_folder/resized)
- `-w, --width`: Target width in pixels (default: 800)
- `-h, --height`: Target height in pixels (default: 600)
- `-q, --quality`: JPEG quality 1-100 (default: 85)
- `-f, --format`: Convert to format (jpg, png, webp, bmp)
- `--no-aspect`: Don't maintain aspect ratio

## Examples

### Example 1: Basic Resize
```bash
python image_resizer.py ./photos
```
- Resizes all images in `./photos` folder
- Output saved to `./photos/resized`
- Target size: 800x600 (maintaining aspect ratio)

### Example 2: High-Quality Thumbnails
```bash
python image_resizer.py ./images -w 300 -h 300 -q 95 -f jpg
```
- Creates 300x300 thumbnails
- High quality JPEG output
- Converts all formats to JPG

### Example 3: Web Optimization
```bash
python image_resizer.py ./website-images -w 1200 -h 800 -f webp -o ./optimized
```
- Resizes for web use
- Converts to WebP format
- Saves to custom output folder

## Supported Formats

**Input**: JPG, JPEG, PNG, BMP, TIFF, WebP
**Output**: JPG, PNG, WebP, BMP

## Features in Detail

### Aspect Ratio Preservation
By default, the tool maintains the original aspect ratio of images. It calculates the best fit within your specified dimensions.

### Format Conversion
Automatically handles format conversion with proper color space management (RGBA to RGB for JPEG).

### Error Handling
Gracefully handles corrupted files and unsupported formats with detailed error messages.

### Progress Tracking
Shows real-time progress with before/after dimensions and success/failure counts.

## Sample Output
```
Found 5 image(s) to process...
Output folder: ./photos/resized
Target size: 800x600 (maintain aspect: True)
------------------------------------------------------------
✓ Resized: photo1.jpg -> photo1_resized.jpg (3024x4032 -> 450x600)
✓ Resized: photo2.png -> photo2_resized.png (1920x1080 -> 800x450)
✓ Resized: photo3.jpg -> photo3_resized.jpg (2048x1536 -> 800x600)
✗ Error processing corrupted.jpg: cannot identify image file
✓ Resized: photo4.webp -> photo4_resized.webp (1024x768 -> 800x600)
------------------------------------------------------------
Processing complete: 4 successful, 1 failed
```

## Use Cases

- **Web Development**: Optimize images for websites
- **Social Media**: Create consistent thumbnail sizes
- **Photography**: Batch process photo collections
- **E-commerce**: Standardize product images
- **Mobile Apps**: Prepare assets for different screen sizes