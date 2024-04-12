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

import logging
import sys

import numpy as np

import openvino as ov

import cv2




logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.DEBUG, stream=sys.stdout)
log = logging.getLogger()

def convert_result_to_image(result,model_name) -> np.ndarray:
    """
    Convert network result of floating point numbers to image with integer
    values from 0-255. Values outside this range are clipped to 0 and 255.

    :param result: a single superresolution network result in N,C,H,W shape
    """
    if "edsr" in model_name:
        
        result = result[0]
        result = result.transpose(1, 2, 0)
        
        result[result < 0] = 0
        result[result > 255] = 255
        result = result.astype(np.uint8)
        return result
        
    result = result.squeeze(0).transpose(1, 2, 0)
    result *= 255
    result[result < 0] = 0
    result[result > 255] = 255
    result = result.astype(np.uint8)
    return result


def run(image, model_path, device, model_name):
    
    if model_name == "edsr":
        h,w = image.shape
    else:
        h,w,_ = image.shape

    core = ov.Core()
    model = core.read_model(model=model_path)

    if "esrgan" in model_name or "edsr" in model_name:
        original_image_key = model.inputs.pop()
      
        
    else:
        original_image_key, bicubic_image_key = model.inputs
      

   
        input_height, _ = list(original_image_key.shape)[2:]
        target_height, _ = list(bicubic_image_key.shape)[2:]
    
        upsample_factor = int(target_height / input_height)

    shapes = {}
    for input_layer in model.inputs:
        if input_layer.names.pop() in ["0","input.1","x.1"]:
            shapes[input_layer] = input_layer.partial_shape
    
            shapes[input_layer][2] = h
            shapes[input_layer][3] = w
        elif input_layer.names.pop() == "1":
            shapes[input_layer] = input_layer.partial_shape
            shapes[input_layer][2] = upsample_factor * h
            shapes[input_layer][3] = upsample_factor * w
        
    model.reshape(shapes)

    compiled_model = core.compile_model(model=model, device_name=device)
    if "esrgan" in model_name or "edsr" in model_name:
        original_image_key = compiled_model.inputs.pop()
    else:
        original_image_key, bicubic_image_key = compiled_model.inputs
    output_key = compiled_model.output(0)

    
    if "edsr" in model_name:
        image = np.expand_dims(image, axis=-1)
    
    input_image_original = np.expand_dims(image.transpose(2, 0, 1), axis=0)
    if "esrgan" in model_name :
        input_image_original = input_image_original / 255.0
    
   

    if "esrgan" in model_name or "edsr" in model_name:
        result = compiled_model(
        {
            original_image_key.any_name: input_image_original,
          
        }
        )[output_key]


    else:

        bicubic_image = cv2.resize(
        src=image, dsize=(w*upsample_factor, h*upsample_factor), interpolation=cv2.INTER_CUBIC)
        input_image_bicubic = np.expand_dims(bicubic_image.transpose(2, 0, 1), axis=0)
        input_image_bicubic.shape
    
        result = compiled_model(
            {
                original_image_key.any_name: input_image_original,
                bicubic_image_key.any_name: input_image_bicubic,
            }
        )[output_key]

    result_image = convert_result_to_image(result,model_name)
    return result_image


if __name__ == "__main__":
    import numpy as np
    img = cv2.imread(r"D:\git\\GIMP-OV\sampleinput\haze.png") #[:, :, ::-1]

    mask = run(img, r"C:\GIMP-OV\weights\superresolution-ov\realesrgan.xml","NPU","esrgan") #r )
    cv2.imwrite("esrgan_ov.png", mask)