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


def run(device,prompt,model_path):
 
     log.info('Initializing Inference Engine...')
  

     scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
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

     image = engine(
        prompt = prompt,
        init_image = None,
        mask = None, 
        strength = 0.5,
        num_inference_steps = 16,
        guidance_scale = 7.5,
        eta = 0.0
    )  



        
     return image


#if __name__ == "__main__":
#    mask = run("GPU", "Dinner under stars","C:\\Users\\lab_admin\\GIMP-OV\\weights\\stable-diffusion-ov")
#    cv2.imwrite("stablediffusion.png", mask)

