"""
 Copyright (C) 2018-2022 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import openvino as ov

# Configure logging
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def convert_result_to_image(result: np.ndarray, model_name: str) -> np.ndarray:
    """
    Convert network result of floating point numbers to image with integer
    values from 0-255. Values outside this range are clipped to 0 and 255.

    :param result: a single superresolution network result in N,C,H,W shape
    :param model_name: name of the model to determine processing steps
    :return: resulting image as np.ndarray
    """
    if "edsr" in model_name:
        result = result[0].transpose(1, 2, 0)
    else:
        result = result.squeeze(0).transpose(1, 2, 0)
        result *= 255

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def validate_inputs(image: np.ndarray, model_path: str, model_name: str) -> bool:
    """Validate inputs before processing."""
    if not os.path.exists(model_path):
        log.error(f"Model file not found: {model_path}")
        return False
    
    if image is None:
        log.error("Input image is None")
        return False
    
    if "edsr" in model_name and len(image.shape) != 2:
        log.error("EDSR model requires grayscale input image")
        return False
    
    return True


def prepare_image_for_model(image: np.ndarray, model_name: str) -> Tuple[np.ndarray, int, int]:
    """Prepare image for model input and return dimensions."""
    if "edsr" in model_name:
        if len(image.shape) == 3:
            # Convert to grayscale if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        image = np.expand_dims(image, axis=-1)
    else:
        h, w, _ = image.shape
    
    return image, h, w


def get_model_inputs(model, model_name: str):
    """Extract model input keys without modifying the original model."""
    inputs = list(model.inputs)
    
    if "esrgan" in model_name or "edsr" in model_name:
        return inputs[0], None, None
    else:
        original_image_key, bicubic_image_key = inputs
        input_height, _ = list(original_image_key.shape)[2:]
        target_height, _ = list(bicubic_image_key.shape)[2:]
        upsample_factor = int(target_height / input_height)
        return original_image_key, bicubic_image_key, upsample_factor


def run(image: np.ndarray, model_path: str, device: str, model_name: str) -> Optional[np.ndarray]:
    """
    Run the superresolution model on the input image.

    :param image: input image as np.ndarray
    :param model_path: path to the model file
    :param device: device to run the inference on
    :param model_name: name of the model to determine processing steps
    :return: super resolution image as np.ndarray or None if failed
    """
    try:
        # Validate inputs
        if not validate_inputs(image, model_path, model_name):
            return None
        
        # Prepare image and get dimensions
        processed_image, h, w = prepare_image_for_model(image, model_name)
        
        # Initialize OpenVINO
        core = ov.Core()
        
        # Set cache directory for ESRGAN on non-GPU devices
        if "esrgan" in model_name and "gpu" not in device.lower():
            cache_dir = os.path.join(os.path.dirname(model_path), 'cache')
            core.set_property({'CACHE_DIR': cache_dir})
        
        # Load model
        log.info(f"Loading model: {model_path}")
        model = core.read_model(model=model_path)
        
        # Get input information
        original_image_key, bicubic_image_key, upsample_factor = get_model_inputs(model, model_name)
        
        # Reshape model for dynamic input sizes
        shapes = {}
        for input_layer in model.inputs:
            layer_name = next(iter(input_layer.names), None)  # Get first name safely
            if layer_name is None:
                continue  # Skip this input_layer if no names are present
            if layer_name in ["0", "input.1", "x.1"]:
                shapes[input_layer] = input_layer.partial_shape
                shapes[input_layer][2] = h
                shapes[input_layer][3] = w
            elif layer_name == "1" and upsample_factor:
                shapes[input_layer] = input_layer.partial_shape
                shapes[input_layer][2] = upsample_factor * h
                shapes[input_layer][3] = upsample_factor * w
        
        if shapes:
            model.reshape(shapes)
        
        config = None
        if device == "NPU":
            config = {"NPU_USE_NPUW": "YES", "NPU_BYPASS_UMD_CACHING": "YES", "NPUW_PARALLEL_COMPILE": "YES", "NPUW_FOLD": "YES"} # compile much faster with NPUW config

        # Compile model with timing
        log.info(f"Compiling model for device: {device}")
        log.debug(f"using config {config}")
        start_time = time.perf_counter()
        compiled_model = core.compile_model(model=model, device_name=device, config=config)
        compilation_time = time.perf_counter() - start_time
        log.info(f"Model compilation completed in {compilation_time:.3f} seconds")
        
        # Get compiled model inputs and outputs
        compiled_inputs = list(compiled_model.inputs)
        if "esrgan" in model_name or "edsr" in model_name:
            original_input = compiled_inputs[0]
            bicubic_input = None
        else:
            original_input, bicubic_input = compiled_inputs
        
        output_key = compiled_model.output(0)
        
        # Prepare input data
        input_image_original = np.expand_dims(processed_image.transpose(2, 0, 1), axis=0)
        if "esrgan" in model_name:
            input_image_original = input_image_original / 255.0
        
        inputs = {original_input.any_name: input_image_original}
        
        # Add bicubic input if needed
        if bicubic_input is not None:
            bicubic_image = cv2.resize(
                src=image, 
                dsize=(w * upsample_factor, h * upsample_factor), 
                interpolation=cv2.INTER_CUBIC
            )
            input_image_bicubic = np.expand_dims(bicubic_image.transpose(2, 0, 1), axis=0)
            inputs[bicubic_input.any_name] = input_image_bicubic
        
        # create inference request.
        infer_request = compiled_model.create_infer_request()

        # Run inference
        log.info("Running inference...")
        start_time = time.perf_counter()
        result = infer_request.infer(inputs)[output_key]
        inference_time = time.perf_counter() - start_time
        log.info(f"Inference completed in {inference_time:.3f} seconds")
        
        # Convert result to image
        result_image = convert_result_to_image(result, model_name)
        return result_image
    
    except Exception as e:
        log.error(f"Error during model execution: {type(e).__name__}: {e}")
        return None
