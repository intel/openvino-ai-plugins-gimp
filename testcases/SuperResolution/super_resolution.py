import argparse
import logging
import os
import sys

import cv2
import numpy as np
import gimpopenvino.plugins.openvino_utils.tools.openvino_common.superes_run_ov as gimp_sr

# Configure logging
log = gimp_sr.logging.getLogger()

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Super resolution using OpenVINO')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('-m', '--model', 
                       choices=['realesrgan-x4-fp16', 'single-image-super-resolution-1032', 'single-image-super-resolution-1033'],
                       default='realesrgan-x4-fp16',
                       help='Model to use for super resolution (default: realesrgan)')
    parser.add_argument('-d', '--device', 
                       choices=['CPU', 'GPU', 'NPU', 'AUTO'],
                       default='NPU',
                       help='Device to run inference on (default: NPU)')
    parser.add_argument('-o', '--output',
                       help='Output filename (default: auto-generated)')
    args = parser.parse_args()
    
    # Check if input image file exists
    if not os.path.exists(args.image_path):
        log.error(f"Input image file does not exist: {args.image_path}")
        return 1
    
    if not os.path.isfile(args.image_path):
        log.error(f"Input path is not a file: {args.image_path}")
        return 1
    
    # Check if input file is readable
    if not os.access(args.image_path, os.R_OK):
        log.error(f"Input image file is not readable: {args.image_path}")
        return 1
    
    # Construct model path
    model_path = os.path.join(
        os.environ.get('USERPROFILE', ''), 
        'openvino-ai-plugins-gimp', 
        'weights', 
        'superresolution-ov', 
        f'{args.model}.xml'
    )
    
    # Check if model file exists
    if not os.path.exists(model_path):
        log.error(f"Model file does not exist: {model_path}")
        log.error("Please ensure the OpenVINO AI plugins are properly installed")
        return 1
    
    if not os.path.isfile(model_path):
        log.error(f"Model path is not a file: {model_path}")
        return 1
    
    if not os.access(model_path, os.R_OK):
        log.error(f"Model file is not readable: {model_path}")
        return 1
    
    # Check for associated model files (.bin file)
    model_bin_path = model_path.replace('.xml', '.bin')
    if not os.path.exists(model_bin_path):
        log.error(f"Model binary file does not exist: {model_bin_path}")
        log.error("OpenVINO models require both .xml and .bin files")
        return 1
    
    # Check if output directory is writable (if output path specified)
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if not os.path.exists(output_dir):
            log.error(f"Output directory does not exist: {output_dir}")
            return 1
        if not os.access(output_dir, os.W_OK):
            log.error(f"Output directory is not writable: {output_dir}")
            return 1
    else:
        # Check if current directory is writable for auto-generated filename
        if not os.access('.', os.W_OK):
            log.error("Current directory is not writable for output file")
            return 1
    
    # Read input image
    log.info(f"Reading input image: {args.image_path}")
    try:
        img = cv2.imread(args.image_path)
        if img is None:
            log.error(f"Failed to read the input image: {args.image_path}")
            log.error("The file may be corrupted or in an unsupported format")
            return 1
    except Exception as e:
        log.error(f"Exception occurred while reading image: {e}")
        return 1
    
    # Run super resolution
    log.info(f"Running super resolution with model: {args.model} on device: {args.device}")
    try:
        result_image = gimp_sr.run(img, model_path, args.device, args.model)
        if result_image is None:
            log.error("Failed to generate the super resolution image.")
            return 1
    except Exception as e:
        log.error(f"Exception occurred during super resolution: {e}")
        return 1
    
    # Save result
    output_filename = args.output or f"{args.model}_{args.device.lower()}_output.png"
    try:
        success = cv2.imwrite(output_filename, result_image)
        if not success:
            log.error(f"Failed to save output image: {output_filename}")
            return 1
        log.info(f"Super resolution image saved as: {output_filename}")
    except Exception as e:
        log.error(f"Exception occurred while saving output: {e}")
        return 1
    
    return 0

    
if __name__ == "__main__":
    sys.exit(main())