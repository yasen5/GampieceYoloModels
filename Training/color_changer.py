#!/usr/bin/env python3
"""
Color replacement script that converts shades of a source color to corresponding
shades of a target color in an image or directory of images.

Usage:
    python3 color_replace.py input.png --src=red --target=yellow --output=output.png
    python3 color_replace.py input_dir/ --src=red --target=yellow --output=output_dir/
"""

import argparse
import numpy as np
from PIL import Image
import colorsys
import os
from pathlib import Path


def show_comparison(original_img, transformed_img):
    """Display original and transformed images side-by-side."""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(original_img)
        ax1.set_title('Original', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(transformed_img)
        ax2.set_title('Transformed', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nWarning: matplotlib not installed. Cannot display comparison.")
        print("Install with: pip install matplotlib")


def parse_color(color_str):
    """Parse color string to RGB tuple (0-255 range)."""
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'gray': (128, 128, 128),
        'grey': (128, 128, 128),
    }
    
    color_str = color_str.lower()
    
    # Check if it's a named color
    if color_str in color_map:
        return color_map[color_str]
    
    # Check if it's a hex color
    if color_str.startswith('#'):
        color_str = color_str[1:]
    if len(color_str) == 6:
        try:
            return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    
    # Try parsing as RGB tuple
    if ',' in color_str:
        try:
            rgb = tuple(int(x.strip()) for x in color_str.split(','))
            if len(rgb) == 3 and all(0 <= x <= 255 for x in rgb):
                return rgb
        except ValueError:
            pass
    
    raise ValueError(f"Invalid color format: {color_str}")


def rgb_to_hsv(r, g, b):
    """Convert RGB (0-255) to HSV (H: 0-360, S: 0-1, V: 0-1)."""
    return colorsys.rgb_to_hsv(r/255, g/255, b/255)


def hsv_to_rgb(h, s, v):
    """Convert HSV (H: 0-1, S: 0-1, V: 0-1) to RGB (0-255)."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def replace_color(img_array, src_rgb, target_rgb, tolerance=30):
    """
    Replace shades of source color with corresponding shades of target color.
    
    Args:
        img_array: numpy array of image (H, W, 3 or 4)
        src_rgb: source color as RGB tuple (0-255)
        target_rgb: target color as RGB tuple (0-255)
        tolerance: color matching tolerance (0-100), higher = more permissive
    
    Returns:
        Modified image array
    """
    # Convert to float for processing
    result = img_array.astype(float)
    has_alpha = img_array.shape[2] == 4
    
    # Get HSV values for source and target
    src_h, src_s, src_v = rgb_to_hsv(*src_rgb)
    target_h, target_s, target_v = rgb_to_hsv(*target_rgb)
    
    # Convert image to HSV
    rgb_img = result[:, :, :3] / 255.0
    h = np.zeros(rgb_img.shape[:2])
    s = np.zeros(rgb_img.shape[:2])
    v = np.zeros(rgb_img.shape[:2])
    
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            h[i, j], s[i, j], v[i, j] = colorsys.rgb_to_hsv(*rgb_img[i, j])
    
    # Create mask for pixels matching source color
    # Match based on hue primarily, with some saturation consideration
    hue_diff = np.minimum(np.abs(h - src_h), 1 - np.abs(h - src_h))  # Circular distance
    hue_tolerance = tolerance / 360.0
    
    # For grayscale source colors (low saturation), match all hues
    if src_s < 0.1:
        mask = s < 0.1  # Match other grayscale pixels
    else:
        mask = (hue_diff < hue_tolerance) & (s > 0.1)
    
    # For matching pixels, preserve their brightness and saturation relative to source
    # but change the hue to target
    if target_s < 0.1:  # Target is grayscale
        s[mask] = 0
        h[mask] = 0
    else:
        h[mask] = target_h
        # Optionally adjust saturation towards target
        s[mask] = np.clip(s[mask] * (target_s / max(src_s, 0.1)), 0, 1)
    
    # Convert back to RGB
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            r, g, b = colorsys.hsv_to_rgb(h[i, j], s[i, j], v[i, j])
            result[i, j, 0] = r * 255
            result[i, j, 1] = g * 255
            result[i, j, 2] = b * 255
    
    return result.astype(np.uint8)


def process_image(input_path, output_path, src_color, target_color, tolerance, show_compare=False):
    """Process a single image file."""
    # Load image
    try:
        img = Image.open(input_path)
        img = img.convert('RGBA')  # Ensure we have alpha channel
    except Exception as e:
        print(f"Error loading image {input_path}: {e}")
        return False
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Replace colors
    result_array = replace_color(img_array, src_color, target_color, tolerance)
    
    # Convert back to image
    result_img = Image.fromarray(result_array, 'RGBA')
    
    # Always save as PNG to preserve quality and support transparency
    output_path_str = str(output_path)
    if not output_path_str.lower().endswith('.png'):
        # Replace extension with .png
        output_path_str = str(Path(output_path_str).with_suffix('.png'))
    
    # Save output
    result_img.save(output_path_str)
    print(f"Processed: {input_path} -> {output_path}")
    
    # Show comparison if requested
    if show_compare:
        show_comparison(img, result_img)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Replace shades of a source color with a target color in an image or directory'
    )
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('--src', required=True, help='Source color (e.g., red, #FF0000, 255,0,0)')
    parser.add_argument('--target', required=True, help='Target color (e.g., yellow, #FFFF00, 255,255,0)')
    parser.add_argument('--output', '-o', help='Output image file or directory (default: output.png or output_dir/)')
    parser.add_argument('--tolerance', '-t', type=int, default=30, 
                       help='Color matching tolerance 0-100 (default: 30)')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Show original and transformed images side-by-side (only for single images)')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'],
                       help='Image file extensions to process in directory mode (default: .png .jpg .jpeg .bmp .gif .tiff)')
    
    args = parser.parse_args()
    
    # Parse colors
    try:
        src_color = parse_color(args.src)
        target_color = parse_color(args.target)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Replacing {args.src} {src_color} with {args.target} {target_color}...")
    
    input_path = Path(args.input)
    
    # Check if input is a directory
    if input_path.is_dir():
        # Directory mode
        output_dir = Path(args.output) if args.output else Path('output_dir')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in args.extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {input_path}")
            print(f"Looking for extensions: {args.extensions}")
            return 1
        
        print(f"Found {len(image_files)} image(s) to process")
        
        success_count = 0
        for img_file in image_files:
            output_file = output_dir / img_file.name
            if process_image(img_file, output_file, src_color, target_color, args.tolerance):
                success_count += 1
        
        print(f"\nCompleted: {success_count}/{len(image_files)} images processed successfully")
        print(f"Output saved to: {output_dir}")
        
    else:
        # Single file mode
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return 1
        
        output_path = args.output or 'output.png'
        
        if process_image(input_path, output_path, src_color, target_color, 
                        args.tolerance, args.compare):
            return 0
        else:
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
