"""
Copyright(C) 2022-2023 Intel Corporation
SPDX - License - Identifier: Apache - 2.0

"""

import openvino_genai
import numpy as np

from PIL import Image

import os
from typing import Optional, List


class StableDiffusionEngineGenai:
    def __init__(self, model: str, model_name: str, device: List = ["GPU", "GPU", "GPU"]):
        self.device = device
        ov_cache_dir = os.path.join(model, 'cache')
        self.pipe = openvino_genai.Text2ImagePipeline(model)
        properties = {"CACHE_DIR": ov_cache_dir}
        if model_name == "sdxl_turbo_square":
            self.pipe.reshape(1, 512, 512, 0) 
        elif model_name == "sd_3.5_med_turbo_square":
            self.pipe.reshape(1, 512, 512, 0.5) 
        elif model_name == "sdxl_base_1.0_square":
            self.pipe.reshape(1, 1024, 1024, self.pipe.get_generation_config().guidance_scale)
        else:
             self.pipe.reshape(1, 512, 512, self.pipe.get_generation_config().guidance_scale)
        self.pipe.compile(device[0], device[1], device[2], config=properties)
        
    def __call__(
            self,
            prompt,
            negative_prompt=None,
            num_inference_steps = 32,
            guidance_scale = 7.5,
            seed: Optional[int] = None,
            callback = None,
            callback_userdata = None
    ):
        print(f"Running Stable Diffusion with prompt: {prompt}")

        def callback_genai(step, num_inference_steps, latent):
            if callback:
                callback(step, callback_userdata)
            return False
                
        if negative_prompt is None:
            image_tensor = self.pipe.generate(prompt,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,rng_seed=seed,callback=callback_genai)
        else:
            image_tensor = self.pipe.generate(prompt,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,negative_prompt=negative_prompt,rng_seed=seed,callback=callback_genai)

        return Image.fromarray(image_tensor.data[0])
    

    
    
    
    

