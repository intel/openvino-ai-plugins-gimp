# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

import os
import json
import sys
import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)



import cv2
import torch

import traceback

import logging

from pathlib import Path
from time import perf_counter

import random

from PIL import Image

# scheduler
from diffusers.schedulers import LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler,EulerAncestralDiscreteScheduler,DDIMScheduler
# utils 
import numpy as np
from gimpopenvino.tools.tools_utils import get_weight_path

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common")
sys.path.extend([plugin_loc])

from  models_ov.stable_diffusion_engine_NEW import StableDiffusionEngine

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.DEBUG, stream=sys.stdout)
log = logging.getLogger()




def run(model_name,device_name): 
    weight_path = get_weight_path()
    blobs = False
    #log.info('Loading config file...')

    import json

    # Opening JSON file
    #config_path = os.path.join(weight_path, "stable-diffusion-ov/config.json")
    #f = open(config_path)
    #data = json.load(f)
   
   
    #model_name = data["model_version"] # "SD_1.5" #For SD 1.4 version use "SD_1.4"
    #device_name = data["device_list"] # ["CPU","GPU.0","GPU.0"] #To run on iGPU change to "GPU". To run on dGPU change to "GPU.1"
   
    if model_name == "SD_1.4":
        model_path = os.path.join(weight_path, "stable-diffusion-ov/stable-diffusion-1.4")
        #device_name = ["CPU","GPU.1","GPU.1"]
    elif model_name == "SD_1.5": 
        model_path = os.path.join(weight_path, "stable-diffusion-ov/stable-diffusion-1.5")
        #device_name = ["CPU","GPU.0","GPU.0"]   
    else:
        model_path = os.path.join(weight_path, "stable-diffusion-ov/stable-diffusion-1.4")
        device_name = ["CPU","GPU","GPU"]


    log.info('Initializing Inference Engine...')




    #print("weight_path in run ",model_path)
 

    engine = StableDiffusionEngine(
        model = model_path,
        device = device_name
    )    
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("Ready")
        while True:
            conn, addr = s.accept()
            with conn:
                #print(f"Connected by {addr}")
                while True:
                    #print("Waiting")
                    data = conn.recv(1024)
                    
                    if data.decode() == "kill":
                        sys.exit()
                    print("Waiting")
                    if not data:
                        break
                    try:    
                        weight_path = get_weight_path()
                        with open(os.path.join(weight_path, "..", "gimp_openvino_run_sd.json"), "r") as file:
                            data_output = json.load(file)
                       
                        prompt = data_output["prompt"]
                        scheduler = data_output["scheduler"]
                        negative_prompt = data_output["negative_prompt"]
                        init_image = data_output["initial_image"]
                        num_infer_steps = data_output["num_infer_steps"]
                        guidance_scale = data_output["guidance_scale"]
                        strength = data_output["strength"]
                        seed = data_output["seed"]
                        create_gif = data_output["create_gif"]
                        
                        
                        if scheduler == "LMSDiscreteScheduler":
                             log.info('LMSDiscreteScheduler...')
                             scheduler = LMSDiscreteScheduler(
                                    beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    
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
              
                        
                        strength = 1.0 if init_image is None else strength
                        log.info('Starting inference...')
                        log.info('Prompt: %s',prompt)
                        log.info('negative_prompt: %s',negative_prompt)
                        log.info('num_inference_steps: %s',num_infer_steps)
                        log.info('guidance_scale: %s',guidance_scale)
                        log.info('strength: %s',strength)
                        log.info('init_image: %s',init_image)
                        
                        if seed is not None:   
                            np.random.seed(int(seed))
                            log.info('Seed: %s',seed)
                        else:
                            ran_seed = random.randrange(4294967294) #4294967294 
                            np.random.seed(int(ran_seed))
                            log.info('Random Seed: %s',ran_seed)
                        
                        
                        output = engine(
                            prompt = prompt,
                            negative_prompt = negative_prompt,
                            init_image = None if init_image is None else Image.open(init_image), #cv2.imread(init_image),
                            scheduler = scheduler,
                            strength = strength,
                            num_inference_steps = num_infer_steps,
                            guidance_scale = guidance_scale,
                            eta = 0.0,
                            create_gif = bool(create_gif),
                            model = model_path
                        ) 
                        
                  
                        cv2.imwrite(os.path.join(weight_path, "..", "cache.png"), output) #, output[:, :, ::-1])
                      
                        src_height,src_width, _ = output.shape
                        data_output["src_height"] = src_height
                        data_output["src_width"] = src_width

                        data_output["inference_status"] = "success"
                        with open(os.path.join(weight_path, "..", "gimp_openvino_run_sd.json"), "w") as file:
                            json.dump(data_output, file)

                        # Remove old temporary error files that were saved
                        my_dir = os.path.join(weight_path, "..")
                        for f_name in os.listdir(my_dir):
                            if f_name.startswith("error_log"):
                                os.remove(os.path.join(my_dir, f_name))

                    except Exception as error:
                        with open(os.path.join(weight_path, "..", "gimp_openvino_run_sd.json"), "w") as file:
                            json.dump({"inference_status": "failed"}, file)
                        with open(os.path.join(weight_path, "..", "error_log.txt"), "w") as file:
                            traceback.print_exception("DEBUG THE ERROR", file=file)                            

                    conn.sendall(data)      
                

if __name__ == "__main__":
   
   
   
   model_name = sys.argv[1]
   device_ = sys.argv[2]

   device_name = [device_,device_,device_]   
   run(model_name,device_name)
   
   print("Exiting Run")
   