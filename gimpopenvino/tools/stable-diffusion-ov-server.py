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

import datetime
import shutil

from PIL import Image

# scheduler
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, LCMScheduler,EulerDiscreteScheduler
# utils 
import numpy as np
from gimpopenvino.tools.tools_utils import get_weight_path

import psutil
import threading

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common")
sys.path.extend([plugin_loc])

from models_ov.stable_diffusion_engine import StableDiffusionEngineAdvanced, StableDiffusionEngine, LatentConsistencyEngine, StableDiffusionEngineReferenceOnly
from models_ov.stable_diffusion_engine_inpainting import StableDiffusionEngineInpainting
from models_ov.stable_diffusion_engine_inpainting_advanced import StableDiffusionEngineInpaintingAdvanced

from  models_ov.controlnet_openpose import ControlNetOpenPose
from  models_ov.controlnet_canny_edge import ControlNetCannyEdge
from  models_ov.controlnet_scribble import ControlNetScribble, ControlNetScribbleAdvanced
from  models_ov.controlnet_openpose_advanced import ControlNetOpenPoseAdvanced
from  models_ov.controlnet_cannyedge_advanced import ControlNetCannyEdgeAdvanced

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.DEBUG, stream=sys.stdout)
log = logging.getLogger()

def progress_callback(i, conn):
    tosend = bytes(str(i), 'utf-8')
    conn.sendall(tosend)

def run(model_name, available_devices, power_mode):
    print("garth debug - run called on ",model_name," with power_mode =",power_mode)
    weight_path = get_weight_path()
    blobs = False

    scheduler = EulerDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
    )
 
    import json
    log.info('Model Name: %s',model_name )
    if model_name == "SD_1.4":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.4")
    elif model_name == "SD_1.5_square_lcm":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5", "square_lcm")    
    elif model_name == "SD_1.5_portrait":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5", "portrait")
    elif model_name == "SD_1.5_square":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5", "square")
    elif model_name == "SD_1.5_square_int8":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5", "square_int8")
        blobs = True
        swap = True
    elif model_name == "SD_1.5_landscape":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5", "landscape")
    elif model_name == "SD_1.5_portrait_512x768":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5", "portrait_512x768")
    elif model_name == "SD_1.5_landscape_768x512":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5", "landscape_768x512")
    elif model_name == "SD_1.5_Inpainting":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5-inpainting")
    elif model_name == "SD_1.5_Inpainting_int8":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5-inpainting-int8")
        blobs = True
    elif model_name == "SD_2.1_square_base":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-2.1", "square_base")
    elif model_name == "SD_2.1_square":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-2.1", "square")
    elif model_name == "controlnet_referenceonly":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-referenceonly")
    elif model_name == "controlnet_openpose":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-openpose")
    elif model_name == "controlnet_canny":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-canny")
    elif model_name == "controlnet_scribble": 
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-scribble")
    elif model_name=="controlnet_openpose_int8":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-openpose-int8")
        blobs = True
        swap = True
    elif model_name=="controlnet_canny_int8":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-canny-int8")
        blobs = True
        swap = True
    elif model_name=="controlnet_scribble_int8":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-scribble-int8")
        blobs = True
        swap = True   
    else:
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.4")
        

    log.info('Initializing Inference Engine...')
    log.info('Model Path: %s',model_path )
    device_list = ["CPU","CPU","CPU","CPU"]
    model_config = []
    model_config_file_name = os.path.join(model_path, "config.json")
    try:
        with open(model_config_file_name,'r') as file:
            model_config = json.load(file)
            if model_config['power modes supported'].lower() == "yes":
                device_list = model_config[power_mode.lower()]
            else:
                device_list = model_config['best_performance']

        # if there is a dGPU available, choose that instead of integrated, unless we are trying to save power for some reason. 
        for device in available_devices:
            if isinstance(device, str)  and \
               device.lower() == 'dgpu' and \
               power_mode.lower() != 'best power efficiency':
                device_list = [device.replace('GPU','GPU.1') if isinstance(device, str) else device for device in device_list]

    except KeyError as e:
        log.error(f"Key Error {e}. Only CPU will be used.")
    except FileNotFoundError:
        log.error("Configuration file is unable to be opened. Only CPU will be used.")
    except json.JSONDecodeError:
        log.error("Error decoding JSON from config.json")        

    if model_name == "SD_1.5_square_int8":
        log.info('device_name: %s',device_list)
        engine = StableDiffusionEngineAdvanced(
        model = model_path,
        device = device_list, 
        blobs = blobs,
        swap = swap)

    elif model_name == "controlnet_openpose_int8":
        log.info('device_name: %s',device_list)
        engine = ControlNetOpenPoseAdvanced(
        model = model_path,
        device = device_list, 
        blobs = blobs,
        swap = swap)

    elif model_name == "controlnet_canny_int8":
        log.info('device_name: %s',device_list)
        engine = ControlNetCannyEdgeAdvanced(
        model = model_path,
        device = device_list, 
        blobs = blobs,
        swap = swap)

    elif model_name == "controlnet_scribble_int8":
        log.info('device_name: %s',device_list)
        engine = ControlNetScribbleAdvanced(
        model = model_path,
        device = device_list, 
        blobs = blobs,
        swap = swap)

    elif model_name ==  "SD_1.5_Inpainting":
        engine = StableDiffusionEngineInpainting(
        model = model_path,
        device= device_list
    )
    
    elif model_name == "controlnet_canny":
        engine = ControlNetCannyEdge(
        model = model_path,
        device= device_list
    )    
    
    elif model_name == "controlnet_scribble":
        engine = ControlNetScribble(
        model = model_path,
        device= device_list
    )

    elif model_name ==  "SD_1.5_square_lcm":
        # device = ["CPU","NPU","GPU"]  
        engine = LatentConsistencyEngine(
        model = model_path,
        device= device_list
    )

    elif model_name == "SD_1.5_Inpainting_int8":
        log.info('advanced Inpainting device_name: %s',device_list)
        engine = StableDiffusionEngineInpaintingAdvanced(
        model = model_path,
        device = device_list, 
        blobs = blobs
        )

    elif model_name == "controlnet_openpose":
        engine = ControlNetOpenPose(
        model = model_path,
        device= device_list
        )
    
    elif model_name == "controlnet_referenceonly":
        engine = StableDiffusionEngineReferenceOnly(
        model = model_path,
        device = ["CPU", "GPU", "GPU", "GPU"]
        )

    else:
        engine = StableDiffusionEngine(
            model = model_path,
            device= device_list
        )


    with (socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s):
        s.bind((HOST, PORT))
        s.listen()
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.connect((HOST, 65433))
        s2.sendall(b"Ready")
        print("Ready")
        while True:
            conn, addr = s.accept()
            with conn:

                while True:
                    print("Waiting")
                    data = conn.recv(1024)

                    if data.decode() == "kill":
                        os._exit(0)
                    if data.decode() == "ping":
                        conn.sendall(data)
                        continue
                    if data.decode() == "model_name":
                        tosend = bytes(model_name, 'utf-8')
                        conn.sendall(tosend)
                        continue

                    if not data:
                        break
                    try:
                        weight_path = get_weight_path()
                        with open(os.path.join(weight_path, "..", "gimp_openvino_run_sd.json"), "r") as file:
                            data_output = json.load(file)

                        prompt = data_output["prompt"]
                        negative_prompt = data_output["negative_prompt"]
                        init_image = data_output["initial_image"]
                        num_images = data_output["num_images"]
                        num_infer_steps = data_output["num_infer_steps"]
                        guidance_scale = data_output["guidance_scale"]
                        strength = data_output["strength"]
                        seed = data_output["seed"]
                        create_gif = False 

                        strength = 1.0 if init_image is None else strength
                        log.info('Starting inference...')
                        log.info('Prompt: %s',prompt)

                        if model_name != "SD_1.5_square_lcm":
                            log.info('negative_prompt: %s',negative_prompt)
                        log.info('num_inference_steps: %s',num_infer_steps)
                        log.info('num_images: %s',num_images)
                        log.info('guidance_scale: %s',guidance_scale)
                        log.info('strength: %s',strength)
                        log.info('init_image: %s',init_image)



                        import time

                        if seed is not None:
                            np.random.seed(int(seed))
                            log.info('Seed: %s',seed)
                        else:
                            seed = random.randrange(4294967294) #4294967294
                            np.random.seed(int(seed))
                            log.info('Random Seed: %s',seed)      
                        
                        start_time = time.time()                      

                        if model_name ==  "SD_1.5_Inpainting" or model_name == "SD_1.5_Inpainting_int8":
                       
                            output = engine(
                                prompt = prompt,
                                negative_prompt = negative_prompt,
                                image = Image.open(os.path.join(weight_path, "..", "cache1.png")),
                                mask_image = Image.open(os.path.join(weight_path, "..", "cache0.png")),
                                scheduler = scheduler,
                                strength = strength,
                                num_inference_steps = num_infer_steps,
                                guidance_scale = guidance_scale,
                                eta = 0.0,
                                create_gif = bool(create_gif),
                                model = model_path,
                                callback = progress_callback,
                                callback_userdata = conn
                        )

                        elif model_name ==  "controlnet_openpose" or model_name == "controlnet_openpose_int8":
                            output = engine(
                                prompt = prompt,
                                negative_prompt = negative_prompt,
                                image = Image.open(init_image),
                                scheduler = scheduler,
                                num_inference_steps = num_infer_steps,
                                guidance_scale = guidance_scale,
                                eta = 0.0,
                                create_gif = bool(create_gif),
                                model = model_path,
                                callback = progress_callback,
                                callback_userdata = conn
                        )
                        elif model_name ==  "controlnet_canny" or model_name == "controlnet_canny_int8":
                            output = engine(
                                prompt = prompt,
                                negative_prompt = negative_prompt,
                                image = Image.open(init_image),
                                scheduler = scheduler,
                                num_inference_steps = num_infer_steps,
                                guidance_scale = guidance_scale,
                                eta = 0.0,
                                create_gif = bool(create_gif),
                                model = model_path,
                                callback = progress_callback,
                                callback_userdata = conn
                        )
                        elif model_name == "SD_1.5_square_lcm":
                            scheduler = LCMScheduler(
                                beta_start=0.00085,
                                beta_end=0.012,
                                beta_schedule="scaled_linear"
                                )
                            output = engine(
                                prompt = prompt,
                                num_inference_steps = num_infer_steps,
                                guidance_scale = guidance_scale,
                                scheduler = scheduler,
                                lcm_origin_steps = 50,
                                model = model_path,
                                callback = progress_callback,
                                callback_userdata = conn,
                                seed = seed
                        )
                        elif model_name == "controlnet_scribble" or model_name == "controlnet_scribble_int8":
                            output = engine(
                                prompt = prompt,
                                negative_prompt = negative_prompt,
                                image = Image.open(init_image),
                                scheduler = scheduler,
                                num_inference_steps = num_infer_steps,
                                guidance_scale = guidance_scale,
                                eta = 0.0,
                                create_gif = bool(create_gif),
                                model = model_path,
                                callback = progress_callback,
                                callback_userdata = conn
                        )          
                        elif model_name == "controlnet_referenceonly":
                            output = engine(
                                prompt = prompt,
                                negative_prompt = negative_prompt,
                                init_image = Image.open(init_image),
                                scheduler = scheduler,
                                num_inference_steps = num_infer_steps,
                                guidance_scale = guidance_scale,
                                eta = 0.0,
                                create_gif = bool(create_gif),
                                model = model_path,
                                callback = progress_callback,
                                callback_userdata = conn
                        )          
                        else:
                            if model_name == "SD_2.1_square":
                                scheduler = EulerDiscreteScheduler(
                                            beta_start=0.00085,
                                            beta_end=0.012,
                                            beta_schedule="scaled_linear",
                                            prediction_type = "v_prediction") 
                            model = model_path
                            if "SD_2.1" in model_name:
                                model = model_name
                            
                            output = engine(
                                prompt = prompt,
                                negative_prompt = negative_prompt,
                                init_image = None if init_image is None else Image.open(init_image),
                                scheduler = scheduler,
                                strength = strength,
                                num_inference_steps = num_infer_steps,
                                guidance_scale = guidance_scale,
                                eta = 0.0,
                                create_gif = bool(create_gif),
                                model = model,
                                callback = progress_callback,
                                callback_userdata = conn
                            )
                        end_time = time.time()
                        print("Image generated from Stable-Diffusion in ", end_time - start_time, " seconds.")

                        image = "sd_cache.png"

                        if model_name == "SD_1.5_square_lcm" or \
                        model_name == "controlnet_openpose" or \
                        model_name == "controlnet_openpose_int8" or \
                        model_name == "controlnet_canny_int8" or \
                        model_name == "controlnet_canny" or \
                        model_name == "controlnet_scribble" or \
                        model_name == "controlnet_scribble_int8":
                            
                            output.save(os.path.join(weight_path, "..", image )) 
                        
                            src_width,src_height = output.size
                        else:
                            cv2.imwrite(os.path.join(weight_path, "..", image), output) #, output[:, :, ::-1])
                    
                            src_height,src_width, _ = output.shape

                        
                       
                        data_output["seed_num"] = seed
                    


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
                            data_output["inference_status"] = "failed"
                            json.dump(data_output, file)
                        with open(os.path.join(weight_path, "..", "error_log.txt"), "w") as file:
                            traceback.print_exception("DEBUG THE ERROR", file=file)

                    conn.sendall(b"done")


def start():
    #
    # args: model_name, supported_devices, device_power_mode
    # 
    #
    model_name = sys.argv[1]
    device_list = sys.argv[2]
    power_mode = sys.argv[3]

    run_thread = threading.Thread(target=run, args=(model_name, device_list, power_mode))
    run_thread.start()

    gimp_proc = None
    for proc in psutil.process_iter():
        if "gimp-2.99" in proc.name():
            gimp_proc = proc
            break;
    
    if gimp_proc:
        psutil.wait_procs([proc])
        print("exiting..!")
        os._exit(0)

    run_thread.join()

if __name__ == "__main__":
   start()


