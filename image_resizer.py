#!/usr/bin/env python3
"""
Image Resizer Tool - Day 7
Batch resize and convert images in a folder
"""

import os
import sys
from PIL import Image
import argparse
from pathlib import Path

class ImageResizer:
    def __init__(self, input_folder, output_folder=None, width=800, height=600, 
                 quality=85, format_type=None, maintain_aspect=True):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder) if output_folder else self.input_folder / "resized"
        self.width = width
        self.height = height
        self.quality = quality
        self.format_type = format_type
        self.maintain_aspect = maintain_aspect
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def create_output_folder(self):
        """Create output folder if it doesn't exist"""
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
    def get_image_files(self):
        """Get all image files from input folder"""
        image_files = []
        for file_path in self.input_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        return image_files
        
    def calculate_new_size(self, original_width, original_height):
        """Calculate new size maintaining aspect ratio if needed"""
        if not self.maintain_aspect:
            return (self.width, self.height)
            
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        # Determine new size based on aspect ratio
        if aspect_ratio > 1:  # Landscape
            new_width = self.width
            new_height = int(self.width / aspect_ratio)
        else:  # Portrait or square
            new_height = self.height
            new_width = int(self.height * aspect_ratio)
            
        return (new_width, new_height)
        
    def resize_image(self, image_path):
        """Resize a single image"""
        try:
            with Image.open(image_path) as img:
                # Get original dimensions
                original_width, original_height = img.size
                
                # Calculate new size
                new_size = self.calculate_new_size(original_width, original_height)
                
                # Resize image with high-quality resampling
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Determine output format and filename
                if self.format_type:
                    output_format = self.format_type.upper()
                    if output_format == 'JPG':
                        output_format = 'JPEG'
                    extension = f".{self.format_type.lower()}"
                else:
                    output_format = img.format or 'JPEG'
                    extension = image_path.suffix
                
                # Create output filename
                output_filename = f"{image_path.stem}_resized{extension}"
                output_path = self.output_folder / output_filename
                
                # Convert RGBA to RGB for JPEG format
                if output_format == 'JPEG' and resized_img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', resized_img.size, (255, 255, 255))
                    if resized_img.mode == 'P':
                        resized_img = resized_img.convert('RGBA')
                    rgb_img.paste(resized_img, mask=resized_img.split()[-1] if resized_img.mode == 'RGBA' else None)
                    resized_img = rgb_img
                
                # Save resized image
                save_kwargs = {'format': output_format}
                if output_format == 'JPEG':
                    save_kwargs['quality'] = self.quality
                    save_kwargs['optimize'] = True
                
                resized_img.save(output_path, **save_kwargs)
                
                print(f"✓ Resized: {image_path.name} -> {output_filename} ({original_width}x{original_height} -> {new_size[0]}x{new_size[1]})")
                return True
                
        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {str(e)}")
            return False
            
    def resize_all_images(self):
        """Resize all images in the input folder"""
        # Create output folder
        self.create_output_folder()
        
        # Get all image files
        image_files = self.get_image_files()
        
        if not image_files:
            print(f"No supported image files found in {self.input_folder}")
            return
            
        print(f"Found {len(image_files)} image(s) to process...")
        print(f"Output folder: {self.output_folder}")
        print(f"Target size: {self.width}x{self.height} (maintain aspect: {self.maintain_aspect})")
        print("-" * 60)
        
        # Process each image
        successful = 0
        failed = 0
        
        for image_file in image_files:
            if self.resize_image(image_file):
                successful += 1
            else:
                failed += 1
                
        print("-" * 60)
        print(f"Processing complete: {successful} successful, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Batch resize and convert images")
    parser.add_argument("input_folder", help="Input folder containing images")
    parser.add_argument("-o", "--output", help="Output folder (default: input_folder/resized)")
    parser.add_argument("-w", "--width", type=int, default=800, help="Target width (default: 800)")
    parser.add_argument("-h", "--height", type=int, default=600, help="Target height (default: 600)")
    parser.add_argument("-q", "--quality", type=int, default=85, help="JPEG quality 1-100 (default: 85)")
    parser.add_argument("-f", "--format", choices=['jpg', 'png', 'webp', 'bmp'], help="Convert to format")
    parser.add_argument("--no-aspect", action="store_true", help="Don't maintain aspect ratio")
    
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
        
    # Create resizer instance
    resizer = ImageResizer(
        input_folder=args.input_folder,
        output_folder=args.output,
        width=args.width,
        height=args.height,
        quality=args.quality,
        format_type=args.format,
        maintain_aspect=not args.no_aspect
    )
    
    # Resize all images
    resizer.resize_all_images()


if __name__ == "__main__":
    main()