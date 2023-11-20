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
from diffusers import LCMScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler,EulerAncestralDiscreteScheduler,DDIMScheduler, UniPCMultistepScheduler
# utils 
import numpy as np
from gimpopenvino.tools.tools_utils import get_weight_path

import psutil
import threading

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common")
sys.path.extend([plugin_loc])

from models_ov.stable_diffusion_engine import StableDiffusionEngineAdvanced, StableDiffusionEngine, LatentConsistencyEngine

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


def run(model_name,device_name):
    weight_path = get_weight_path()
    blobs = False
    #log.info('Loading config file...')
    import json
    log.info('Model Name: %s',model_name )
    if model_name == "SD_1.4":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.4")
    elif model_name == "SD_1.5_lcm":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5", "lcm")
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
    elif model_name == "controlnet_openpose":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-openpose")
    elif model_name == "controlnet_canny":
        model_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-canny")
    elif model_name == "controlnet_scribble": 
        model_path = os.path.join(weight_path, "stable-diffusion-ov/controlnet-scribble")
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
        device_name = ["CPU","GPU","GPU"]


    log.info('Initializing Inference Engine...')
    log.info('Model Path: %s',model_path )
    log.info('device_name: %s',device_name)


    if model_name == "SD_1.5_square_int8":
        log.info('device_name: %s',device_name)
        engine = StableDiffusionEngineAdvanced(
        model = model_path,
        device = [device_name[0], device_name[1],device_name[2],device_name[3]],
        blobs = blobs,
        swap = swap)

    elif model_name == "controlnet_openpose_int8":
        log.info('device_name: %s',device_name)
        engine = ControlNetOpenPoseAdvanced(
        model = model_path,
        device = [device_name[0], device_name[1],device_name[2],device_name[3]],
        blobs = blobs,
        swap = swap)

    elif model_name == "controlnet_canny_int8":
        log.info('device_name: %s',device_name)
        engine = ControlNetCannyEdgeAdvanced(
        model = model_path,
        device = [device_name[0], device_name[1],device_name[2],device_name[3]],
        blobs = blobs,
        swap = swap)

    elif model_name == "controlnet_scribble_int8":
        log.info('device_name: %s',device_name)
        engine = ControlNetScribbleAdvanced(
        model = model_path,
        device = [device_name[0], device_name[1],device_name[2],device_name[3]],
        blobs = blobs,
        swap = swap)

    elif model_name ==  "SD_1.5_Inpainting":
        engine = StableDiffusionEngineInpainting(
        model = model_path,
        device = [device_name[0], device_name[1], device_name[3]]
    )
    
    elif model_name == "controlnet_canny":
        engine = ControlNetCannyEdge(
        model = model_path,
        device = device_name
    )    
    
    elif model_name == "controlnet_scribble":
        engine = ControlNetScribble(
        model = model_path,
        device = device_name
    )

    elif model_name ==  "SD_1.5_lcm":
        engine = LatentConsistencyEngine(
        model = model_path,
        device = [device_name[0], device_name[1], device_name[3]]
    )

    elif model_name == "SD_1.5_Inpainting_int8":
        log.info('advanced Inpainting device_name: %s',device_name)
        engine = StableDiffusionEngineInpaintingAdvanced(
        model = model_path,
        device = [device_name[0], device_name[1],device_name[2],device_name[3]],
        blobs = blobs
        )

    elif model_name == "controlnet_openpose":
        engine = ControlNetOpenPose(
        model = model_path,
        device = [device_name[0], device_name[1], device_name[3]]
        )

    else:
        engine = StableDiffusionEngine(
            model = model_path,
            device = [device_name[0], device_name[1], device_name[3]]
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
                        scheduler = data_output["scheduler"]
                        negative_prompt = data_output["negative_prompt"]
                        init_image = data_output["initial_image"]
                        num_infer_steps = data_output["num_infer_steps"]
                        guidance_scale = data_output["guidance_scale"]
                        strength = data_output["strength"]
                        seed = data_output["seed"]
                        create_gif = False #data_output["create_gif"]

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

                        elif scheduler == "UniPCMultistepScheduler":
                            log.info('UniPCMultistepScheduler')
                            scheduler = UniPCMultistepScheduler(
                                beta_start=0.00085,
                                beta_end=0.012,
                                beta_schedule="scaled_linear"
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

                        import time
                        start_time = time.time()

                        if model_name ==  "SD_1.5_Inpainting" or model_name == "SD_1.5_Inpainting_int8":
                            print("-------In Inpainting-------")
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
                        elif model_name == "SD_1.5_lcm":
                            data_output["scheduler"] = "LCMScheduler"
                            output = engine(
                                prompt = prompt,
                                num_inference_steps = num_infer_steps,
                                guidance_scale = guidance_scale,
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
                        else:
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
                                model = model_path,
                                callback = progress_callback,
                                callback_userdata = conn
                            )
                        end_time = time.time()
                        print("Image generated from Stable-Diffusion in ", end_time - start_time, " seconds.")

                        if model_name == "SD_1.5_lcm" or \
                           model_name == "controlnet_openpose" or \
                           model_name == "controlnet_openpose_int8" or \
                           model_name == "controlnet_canny_int8" or \
                           model_name == "controlnet_canny" or \
                           model_name == "controlnet_scribble" or \
                           model_name == "controlnet_scribble_int8":
                            output.save(os.path.join(weight_path, "..", "cache.png"))
                            src_width,src_height = output.size
                        else:
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
                            data_output["inference_status"] = "failed"
                            json.dump(data_output, file)
                            #json.dump({"inference_status": "failed"}, file)
                        with open(os.path.join(weight_path, "..", "error_log.txt"), "w") as file:
                            traceback.print_exception("DEBUG THE ERROR", file=file)

                    conn.sendall(b"done")


def start():
    model_name = sys.argv[1]
    text_encoder_device = sys.argv[2]
    unet_pos_device = sys.argv[3]
    unet_neg_device = sys.argv[4]
    vae_device = sys.argv[5]

    device_name = [text_encoder_device, unet_pos_device, unet_neg_device, vae_device]
    run_thread = threading.Thread(target=run, args=(model_name, device_name))
    run_thread.start()

    #run(model_name,device_name)
    #print("Looking for GIMP process")
    gimp_proc = None
    for proc in psutil.process_iter():
        if "gimp-2.99" in proc.name():
            gimp_proc = proc
            break;

    #print("Done looking for GIMP process")

    if gimp_proc:
     #   print("gimp-2.99 process found:", gimp_proc)
        psutil.wait_procs([proc])
        print("exiting..!")
        os._exit(0)

    #else:
    #    print("no gimp process found!")

    run_thread.join()


if __name__ == "__main__":
   start()


