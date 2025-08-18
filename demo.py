#!/usr/bin/env python3
"""
Demo script for Image Resizer Tool
Creates sample images and demonstrates the resizer functionality
"""

import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

def create_sample_images():
    """Create sample images for demonstration"""
    sample_folder = Path("sample_images")
    sample_folder.mkdir(exist_ok=True)
    
    # Create different sized sample images
    samples = [
        ("landscape.jpg", (1920, 1080), "JPEG", (70, 130, 180)),
        ("portrait.png", (1080, 1920), "PNG", (220, 20, 60)),
        ("square.jpg", (1024, 1024), "JPEG", (34, 139, 34)),
        ("wide.png", (2560, 1440), "PNG", (255, 140, 0)),
        ("small.jpg", (640, 480), "JPEG", (128, 0, 128))
    ]
    
    print("Creating sample images...")
    
    for filename, size, format_type, color in samples:
        # Create image with solid color
        img = Image.new("RGB", size, color)
        draw = ImageDraw.Draw(img)
        
        # Add some text
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
            
        text = f"{filename}\n{size[0]}x{size[1]}"
        
        # Calculate text position (center)
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = 100, 50  # Approximate
            
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, fill="white", font=font)
        
        # Add some random shapes for visual interest
        for _ in range(5):
            x1, y1 = random.randint(0, size[0]//2), random.randint(0, size[1]//2)
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
            shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], outline=shape_color, width=3)
        
        # Save image
        filepath = sample_folder / filename
        img.save(filepath, format=format_type, quality=90 if format_type == "JPEG" else None)
        print(f"Created: {filename} ({size[0]}x{size[1]})")
    
    print(f"\nSample images created in '{sample_folder}' folder")
    return sample_folder

def run_demo():
    """Run demonstration of the image resizer"""
    from image_resizer import ImageResizer
    
    print("=" * 60)
    print("IMAGE RESIZER TOOL DEMO")
    print("=" * 60)
    
    # Create sample images
    sample_folder = create_sample_images()
    
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Resize (800x600, maintain aspect)")
    print("=" * 60)
    
    # Demo 1: Basic resize
    resizer1 = ImageResizer(
        input_folder=sample_folder,
        output_folder=sample_folder / "demo1_basic",
        width=800,
        height=600,
        maintain_aspect=True
    )
    resizer1.resize_all_images()
    
    print("\n" + "=" * 60)
    print("DEMO 2: Thumbnail Creation (300x300, no aspect ratio)")
    print("=" * 60)
    
    # Demo 2: Thumbnail creation
    resizer2 = ImageResizer(
        input_folder=sample_folder,
        output_folder=sample_folder / "demo2_thumbnails",
        width=300,
        height=300,
        maintain_aspect=False
    )
    resizer2.resize_all_images()
    
    print("\n" + "=" * 60)
    print("DEMO 3: Format Conversion (Convert all to WebP)")
    print("=" * 60)
    
    # Demo 3: Format conversion
    resizer3 = ImageResizer(
        input_folder=sample_folder,
        output_folder=sample_folder / "demo3_webp",
        width=1200,
        height=800,
        format_type="webp",
        maintain_aspect=True
    )
    resizer3.resize_all_images()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("Check the following folders for results:")
    print(f"- {sample_folder / 'demo1_basic'}")
    print(f"- {sample_folder / 'demo2_thumbnails'}")
    print(f"- {sample_folder / 'demo3_webp'}")

if __name__ == "__main__":
    run_demo()