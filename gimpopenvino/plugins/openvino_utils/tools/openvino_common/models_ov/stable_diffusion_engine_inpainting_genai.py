"""
Copyright(C) 2022-2023 Intel Corporation
SPDX - License - Identifier: Apache - 2.0

"""
#from .model import Model


import openvino
import openvino_genai
import numpy as np

from PIL import Image




class StableDiffusionEngineInpaintingGenai:
    def __init__(self, model: str, device: str = "GPU"):
 
        self.device = device
        self.pipe = openvino_genai.InpaintingPipeline(model, device)

    def read_image(self,path: str) -> openvino.Tensor:

        pic = Image.open(path).convert("RGB")
        width, height = pic.size
        print(f"Image loaded: {path} (Width: {width}, Height: {height})")

        # Convert to numpy array (H, W, C) → (1, C, H, W) for OpenVINO
        image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
                
        return openvino.Tensor(image_data)
    


    def __call__(
            self,
            prompt,
            image_path: str = None,
            mask_path: str = None,
            negative_prompt=None,
            scheduler=None,
            strength = 0.5,
            num_inference_steps = 32,
            guidance_scale = 7.5,
            callback = None,
            callback_userdata = None
    ):
        width = 768
        height = 432
        
        image = self.read_image(image_path)
        mask_image = self.read_image(mask_path)


        print(f"Running inpainting with prompt: {prompt}")


        def callback_genai(step, num_inference_steps, latent):
            #print(f"Image generation step: {step} / {num_inference_steps}")
            if callback:
                callback(step, callback_userdata)
            return False
           

        if (image.shape[1] == image.shape[2]):
            image_tensor = self.pipe.generate(prompt, image, mask_image,num_inference_steps=num_inference_steps,negative_prompt=negative_prompt,callback=callback_genai)
        else:
            image_tensor = self.pipe.generate(prompt, image, mask_image, width=width, height=height,num_inference_steps=num_inference_steps,negative_prompt=negative_prompt,callback=callback_genai)
                   

        return Image.fromarray(image_tensor.data[0])
    

    
    
    
    

