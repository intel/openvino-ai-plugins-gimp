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

from pathlib import Path
from time import perf_counter

import os


# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
# utils
import cv2
import numpy as np


#sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
# engine
from  models_ov.stable_diffusion_engine import StableDiffusionEngine
from performance_metrics import PerformanceMetrics

import monitors



logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.DEBUG, stream=sys.stdout)
log = logging.getLogger()


def run(device,prompt,num_infer_steps,guidance_scale,init_image,strength,seed,create_gif,model_path):
 
     log.info('Initializing Inference Engine...')
     if seed is not None:   
        np.random.seed(int(seed))
        log.info('Seed: %s',seed)
  
     if init_image is None:
         log.info('LMSDiscreteScheduler...')
         scheduler = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                tensor_format="np"
            )
     else:
        log.info('PNDMScheduler...')
        scheduler = PNDMScheduler(

            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps = True,
            tensor_format="np"
        ) 


     print("weight_path in run ",model_path)
     log.info('Device: %s',device)
     engine = StableDiffusionEngine(
        model = model_path,
        scheduler = scheduler,
        device = device
    )
     log.info('Starting inference...')
     log.info('Prompt: %s',prompt)
     log.info('num_inference_steps: %s',num_infer_steps)
     log.info('guidance_scale: %s',guidance_scale)
     log.info('strength: %s',strength)

     image = engine(
        prompt = prompt,
        init_image = None if init_image is None else cv2.imread(init_image),
        mask = None, 
        strength = strength,
        num_inference_steps = num_infer_steps,
        guidance_scale = guidance_scale,
        eta = 0.0,
        create_gif = bool(create_gif),
        model = model_path
    )  



        
     return image


#if __name__ == "__main__":
#    mask = run("GPU", "Dinner under stars","C:\\GIMP-OV\\weights\\stable-diffusion-ov")
#    cv2.imwrite("stablediffusion.png", mask)

