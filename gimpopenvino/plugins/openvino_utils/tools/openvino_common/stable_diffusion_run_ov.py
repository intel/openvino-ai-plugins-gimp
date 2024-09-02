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


import os
import random

from PIL import Image

# scheduler
from diffusers.schedulers import LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler
# utils
import cv2
import numpy as np
from sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "tools")])
from tools_utils import get_weight_path import get_weight_path


#sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
# engine
from  models_ov.stable_diffusion_engine_NEW import StableDiffusionEngine





logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.DEBUG, stream=sys.stdout)
log = logging.getLogger()


def run(device,prompt,negative_prompt,num_infer_steps,guidance_scale,init_image,strength,seed,create_gif,scheduler,model_path): #model_path):model_name
     
     weight_path = get_weight_path()
 
     log.info('Initializing Inference Engine...')
     if seed is not None:   
        np.random.seed(int(seed))
        log.info('Seed: %s',seed)
     else:
        ran_seed = random.randrange(4294967294) #4294967294 
        np.random.seed(int(ran_seed))
        log.info('Random Seed: %s',ran_seed)

     if scheduler == "LMSDiscreteScheduler":
         log.info('LMSDiscreteScheduler...')
         scheduler = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear"
            )
     elif scheduler == "PNDMScheduler":
        log.info('PNDMScheduler...')
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps = True,
        ) 
      
     else:
         log.info('EulerDiscreteScheduler...')
         scheduler = EulerDiscreteScheduler(
         beta_start=0.00085, 
         beta_end=0.012, 
         beta_schedule="scaled_linear"
         )


     print("weight_path in run ",model_path)
     log.info('Device: %s',device)
     engine = StableDiffusionEngine(
        model = model_path,
        scheduler = scheduler,
        device = device
    )
     strength = 1.0 if init_image is None else strength
     log.info('Starting inference...')
     log.info('Prompt: %s',prompt)
     log.info('negative_prompt: %s',negative_prompt)
     log.info('num_inference_steps: %s',num_infer_steps)
     log.info('guidance_scale: %s',guidance_scale)
     log.info('strength: %s',strength)
     log.info('init_image: %s',init_image)

     image = engine(
        prompt = prompt,
        negative_prompt = negative_prompt,
        init_image = None if init_image is None else Image.open(init_image), #cv2.imread(init_image),
        mask = None, 
        strength = strength,
        num_inference_steps = num_infer_steps,
        guidance_scale = guidance_scale,
        eta = 0.0,
        create_gif = bool(create_gif),
        model = model_path
    )  



        
     return image


if __name__ == "__main__":
    mask = run("GPU", "photo of a lady in green party dress","ugly, low quality, bad anatomy, monochrome, deformed face", 20, 7.5 ,None,7,None,False,"EulerDiscreteScheduler","sd_1.5") 
    cv2.imwrite("stablediffusion.png", mask)

