#!/usr/bin/env python3
# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from statistics import mean
import platform
import subprocess
import os

import cv2
import numpy as np
from PIL import Image
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, LCMScheduler, EulerDiscreteScheduler
from openvino.runtime import Core

sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common")])
sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","tools")])

from gimpopenvino.plugins.openvino_utils.tools.tools_utils import get_weight_path, SDOptionCache,config_path_dir

from gimpopenvino.plugins.openvino_utils.tools.openvino_common.models_ov import (
    stable_diffusion_engine,
    stable_diffusion_engine_genai,
    stable_diffusion_engine_inpainting_genai,
    stable_diffusion_engine_inpainting,
    stable_diffusion_engine_inpainting_advanced,
    stable_diffusion_3,
    controlnet_openpose,
    controlnet_canny_edge,
    controlnet_scribble,
    controlnet_openpose_advanced,
    controlnet_cannyedge_advanced,
    stable_diffusion_engine_fastsd

)

from gimpopenvino.plugins.openvino_utils.tools.openvino_common.models_ov.fastsd.model_config import ModelConfig
fast_sd_models_config = ModelConfig(os.path.join(config_path_dir, "fastsd_models.json")).load()
fast_sd_models = fast_sd_models_config.get("models", [])
fast_sd_models_up = [model.lower() for model in fast_sd_models]
fast_sd_models_map = {model.lower(): model for model in fast_sd_models}

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()


def get_bios_version():
    try:
        os_name = platform.system()
        
        if os_name == "Windows":
            import wmi
            c = wmi.WMI()
            bios = c.Win32_BIOS()[0]
            return bios.SMBIOSBIOSVersion
        
        elif os_name == "Linux":
            result = subprocess.run(
                ["sudo", "dmidecode", "-t", "bios"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                return f"Error executing dmidecode: {result.stderr.strip()}"
            
            for line in result.stdout.splitlines():
                if "Version:" in line:
                    return line.split("Version:")[1].strip()
            return "BIOS version not found."
        
        else:
            return f"Unsupported OS: {os_name}"
    
    except Exception as e:
        return str(e)

def get_windows_pcie_device_driver_versions():
    import win32com.client
    try:
        # Initialize the WMI client
        wmi = win32com.client.Dispatch("WbemScripting.SWbemLocator")
        service = wmi.ConnectServer(".", "root\\cimv2")
        
        # Query Win32_PnPSignedDriver to get driver information for PCI devices
        drivers = service.ExecQuery("SELECT DeviceID, DriverVersion, Description FROM Win32_PnPSignedDriver")
        
        driver_info_list = []
        for driver in drivers:
            driver_info = {
                "DeviceID": driver.DeviceID,
                "DriverVersion": driver.DriverVersion,
                "Description" : driver.Description
            }
            driver_info_list.append(driver_info)
        
        return driver_info_list
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def check_windows_device_driver_version(device_name, driver_info):
    for info in driver_info:
        if info["Description"] is not None and device_name in info["Description"]:
            return info["DriverVersion"]
    return None

def print_system_info():
    log.info("System Information")
    log.info("==================")
    log.info(f"System: {platform.system()}")
    log.info(f"Node Name: {platform.node()}")
    log.info(f"Python Version: {platform.python_version()}")
    log.info(f"Platform: {platform.platform()}")
    if "window" in platform.system().lower():
        driver_info = get_windows_pcie_device_driver_versions()
        log.info(f'BIOS: {get_bios_version()}')
        log.info(f'NPU Driver: {check_windows_device_driver_version(device_name="AI Boost",driver_info=driver_info)}')
        log.info(f'GPU Driver: {check_windows_device_driver_version(device_name="Arc",driver_info=driver_info)}')  
    elif "linux" in platform.system().lower():
        try:
            log.info(f'BIOS: {get_bios_version()}')
            with open('/sys/module/intel_vpu/version', 'r') as f:
                log.info(f'NPU Driver: {f.readline()}')
            log.info(f'GPU Driver: <unsupported>')
        except:
            pass

def initialize_engine(model_name, model_path, device_list):
    if model_name == "sd_1.5_square_int8":
        return stable_diffusion_engine.StableDiffusionEngineAdvanced(model=model_path, device=device_list)
    if model_name == "sd_3.0_square":
        device_list = ["GPU"]
        return stable_diffusion_3.StableDiffusionThreeEngine(model=model_path, device=device_list)
    if model_name == "sd_1.5_inpainting":
        return stable_diffusion_engine_inpainting_genai.StableDiffusionEngineInpaintingGenai(model=model_path, device=device_list[0])
    if model_name in ("sd_1.5_square_lcm","sdxl_base_1.0_square","sdxl_turbo_square","sd_3.0_med_diffuser_square","sd_3.5_med_turbo_square"):
        return stable_diffusion_engine_genai.StableDiffusionEngineGenai(model=model_path,model_name=model_name,device=device_list)
    if model_name == "sd_1.5_inpainting_int8":
        return stable_diffusion_engine_inpainting_advanced.StableDiffusionEngineInpaintingAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_openpose_int8":
        return controlnet_openpose_advanced.ControlNetOpenPoseAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_canny_int8":
        return controlnet_cannyedge_advanced.ControlNetCannyEdgeAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_scribble_int8":
        return controlnet_scribble.ControlNetScribbleAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_canny":
        return controlnet_canny_edge.ControlNetCannyEdge(model=model_path, device=device_list)
    if model_name == "controlnet_scribble":
        return controlnet_scribble.ControlNetScribble(model=model_path, device=device_list)
    if model_name == "controlnet_openpose":
        return controlnet_openpose.ControlNetOpenPose(model=model_path, device=device_list)
    if model_name == "controlnet_referenceonly":
        return stable_diffusion_engine.StableDiffusionEngineReferenceOnly(model=model_path, device=device_list)
    if model_name in  fast_sd_models:
        return stable_diffusion_engine_fastsd.StableDiffusionEngineFastSD(
            model=model_path, device=device_list, model_name=model_name
        )
    return stable_diffusion_engine.StableDiffusionEngine(model=model_path, device=device_list, model_name=model_name)

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-l', '--list', action = 'store_true', 
                      help='Show list of models currently installed.')
    # base path to models
    args.add_argument('-bp','--model_base_path',type = str, default = None, required = False,
                      help='Optional. Specify the absolute base path to model weights. \nUsage example:  -bp \\stable-diffusion\\model-weights\\')
    # model name
    args.add_argument('-m', '--model_name',type = str, default = "sd_1.5_square_int8", required = False,
                      help='Optional. Model path of directory. Default is sd_1.5_square_int8. \nUsage example:  -m sd_1.5_square_lcm')
    # Target Devices (CPU/GPU/NPU)
    args.add_argument('-td','--text_device',type = str, default = None, required = False,
                      help='Optional. Specify the target device to infer on; CPU, GPU, NPU '
                      'is acceptable for Text encoder. Default value is None.')
    args.add_argument('-ud','--unet_device',type = str, default = None, required = False,
                      help='Optional. Specify the target device to infer on; CPU, GPU, NPU '
                      'is acceptable for Unet. Default value is None.')
    args.add_argument('-und','--unet_neg_device',type = str, default = None, required = False,
                      help='Optional. Specify the target device to infer on; CPU, GPU, NPU '
                      'is acceptable for Unet Negative. Default value is None.')
    args.add_argument('-vd','--vae_device',type = str, default = None, required = False,
                      help='Optional. Specify the target device to infer on; CPU, GPU, NPU '
                      'is acceptable for VAE decoder and encoder. Default value is None.')
    # seed, number of iterations
    args.add_argument('-seed','--seed',type = int, default = 1507302932, required = False,
                      help='Optional. Specify the seed for initialize latent space.')
    args.add_argument('-niter','--iterations',type = int, default = 20, required = False,
                      help='Optional. Iterations for Stable Diffusion.')
    # save output image
    args.add_argument('-si','--save_image',action='store_true', help='Optional. Save output image.')
    
    # generate multiple images
    args.add_argument('-n','--num_images',type = int, default = 1, required = False,
                      help='Optional. Number of images to generate.')
    # guidance scale
    args.add_argument('-g','--guidance_scale',type = float, default = 7.5, required = False,
                      help='Optional. Affects how closely the image prompt is followed.')
    # power mode
    args.add_argument('-pm','--power_mode',type = str, default = "best performance", required = False,
                      help='Optional. Specify the power mode. Default is best performance')
    # prompt, negative prompt
    args.add_argument('-pp','--prompt',type = str, 
                      default ="a portrait of an old coal miner in 19th century, beautiful painting with highly detailed face by greg rutkowski and magali villanueve",
                      required = False,
                      help='Optional. Specify the prompt.  Default: "castle surrounded by water and nature, village, volumetric lighting, photorealistic, detailed and intricate, fantasy, epic cinematic shot, mountains, 8k ultra hd"')
    args.add_argument('-np','--neg_prompt',type = str, default = None , required=False, 
                      help='Optional. Specify the negative prompt.  Default: None')
         
    return parser.parse_args()

def validate_model_paths(base_path: str, model_paths: dict) -> dict:
    """
    Check if model directories exist based on base_path and model_paths structure.

    Args:
        base_path (str): Root directory where models are stored.
        model_paths (dict): Dictionary with model keys and relative path parts.

    Returns:
        dict: Dictionary with model names and a boolean indicating existence.
    """
    results = {}
    for model_name, relative_parts in model_paths.items():
        full_path = os.path.join(base_path, *relative_parts)
        if os.path.isdir(full_path):
            if "int8a16" in model_name:
                if os.path.isfile(os.path.join(full_path, "unet_int8a16.xml")):
                    results[model_name] = full_path
            elif "fp8" in model_name:
                if os.path.isfile(os.path.join(full_path, "unet_fp8.xml")):
                    results[model_name] = full_path
            else:
                results[model_name] = full_path
    return results


def main():
    args = parse_args()
    results = []
    generation_time = []
    use_fastsd = False

    if args.model_base_path:
        weight_path = args.model_base_path
    else:
        weight_path = get_weight_path()
    
    # Check if the directory path exists
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"The directory path {weight_path} does not exist.")
    
    execution_devices = ["GPU"]*5
    
    
    model_name = args.model_name
 
    model_paths = {
        "sd_1.4": ["stable-diffusion-ov", "stable-diffusion-1.4"],
        "sd_1.5_square_lcm": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_lcm"],
        "sdxl_base_1.0_square": ["stable-diffusion-ov", "stable-diffusion-xl", "square_base"],
        "sdxl_turbo_square": ["stable-diffusion-ov", "stable-diffusion-xl", "square_turbo"],
        "sd_1.5_portrait": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait"],
        "sd_1.5_square": ["stable-diffusion-ov", "stable-diffusion-1.5", "square"],
        "sd_1.5_square_int8": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_int8"],
        "sd_1.5_square_int8a16": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_int8"],
        "sd_3.0_med_diffuser_square": ["stable-diffusion-ov", "stable-diffusion-3.0-medium", "square_diffusers" ],
        "sd_3.5_med_turbo_square": ["stable-diffusion-ov", "stable-diffusion-3.5-medium", "square_turbo" ],
        "sd_1.5_landscape": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape"],
        "sd_1.5_portrait_512x768": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait_512x768"],
        "sd_1.5_landscape_768x512": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape_768x512"],
        "sd_1.5_inpainting": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting"],
        "sd_1.5_inpainting_int8": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting_int8"],
        "sd_2.1_square_base": ["stable-diffusion-ov", "stable-diffusion-2.1", "square_base"],
        "sd_2.1_square": ["stable-diffusion-ov", "stable-diffusion-2.1", "square"],
        "sd_3.0_square": ["stable-diffusion-ov", "stable-diffusion-3.0"],
        "controlnet_referenceonly": ["stable-diffusion-ov", "controlnet-referenceonly"],
        "controlnet_openpose": ["stable-diffusion-ov", "controlnet-openpose"],
        "controlnet_canny": ["stable-diffusion-ov", "controlnet-canny"],
        "controlnet_scribble": ["stable-diffusion-ov", "controlnet-scribble"],
        "controlnet_openpose_int8": ["stable-diffusion-ov", "controlnet-openpose-int8"],
        "controlnet_canny_int8": ["stable-diffusion-ov", "controlnet-canny-int8"],
        "controlnet_scribble_int8": ["stable-diffusion-ov", "controlnet-scribble-int8"],
        }
 
    if args.list:
        print(f"\nPre installed models: ")
        for key in validate_model_paths(weight_path, model_paths).keys():
            print(f"{key}")
        print("\n\nFastSD Models (installed at runtime):")
        for fsd_model in fast_sd_models:
            print(f"{fsd_model}")
        exit()

    if model_name not in fast_sd_models_up:
        model_path = os.path.join(weight_path, *model_paths.get(model_name))    
        model_config_file_name = os.path.join(model_path, "config.json")
    else:
        model_path = ""
        use_fastsd = True
    

    try:
        if not use_fastsd and args.power_mode is not None and os.path.exists(model_config_file_name):
            with open(model_config_file_name, 'r') as file:
                model_config = json.load(file)
                if model_config['power modes supported'].lower() == "yes":
                    execution_devices = model_config[args.power_mode.lower()]
                else:
                    execution_devices = model_config['best performance']
    
        # commandline over rides power mode config
        if args.text_device is not None:
            execution_devices[0] = args.text_device
        if args.unet_device is not None:
            execution_devices[1] = args.unet_device
        if args.unet_neg_device is not None:
            execution_devices[2] = args.unet_neg_device
        if args.vae_device is not None:
            execution_devices[3 if "lcm" not in model_name else 2] = args.vae_device

    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        log.error(f"Error loading configuration: {e}. Only CPU will be used.")
        
    print_system_info() 
    
    log.info('')
    log.info('Device : Version')
    core = Core()
    for device in core.available_devices:
        log.info(f'  {device}  : {core.get_versions(device)[device].build_number}')
    log.info('')

    if not use_fastsd:
        log.info('Initializing Inference Engine...') 
        log.info('Model Path: %s',model_path ) 
       
        if "turbo" in model_name and args.guidance_scale > 1.0:
            log.warning(f"Max guidance scale for {model_name} is 1.0, adjusting {args.guidance_scale} down to 1.0")
            args.guidance_scale = 1.0
    else:
        log.info('Initializing FastSD engine...')
    
    prompt = args.prompt #"a beautiful artwork illustration, concept art sketch of an astronaut in white futuristic cybernetic armor in a dark cave, volumetric fog, godrays, high contrast, vibrant colors, vivid colors, high saturation, by Greg Rutkowski and Jesper Ejsing and Raymond Swanland and alena aenami, featured on artstation, wide angle, vertical orientation" 
    negative_prompt = args.neg_prompt # "lowres, bad quality, monochrome, cropped head, deformed face, bad anatomy" 
    
    init_image = None 
    num_infer_steps = args.iterations 
    guidance_scale = args.guidance_scale 
    strength = 1.0
    seed = 4294967294   
    
    scheduler = EulerDiscreteScheduler( 
                    beta_start=0.00085,  
                    beta_end=0.012,  
                    beta_schedule="scaled_linear" 
    ) 
    
    engine = None
    if not use_fastsd:
        engine = initialize_engine(model_name=model_name, model_path=model_path, device_list=execution_devices)
    else:
        engine = initialize_engine(fast_sd_models_map[model_name], model_name, execution_devices)

    current_time = datetime.now()

    # 24-hour format
    timestamp_24 = current_time.strftime("%Y%m%d-%H%M%S")

    for i in range(0,args.num_images):
        log.info('Starting inference...') 
        log.info('Prompt: %s',prompt) 
        log.info('negative_prompt: %s',negative_prompt) 
        log.info('num_inference_steps: %s',num_infer_steps) 
        log.info('guidance_scale: %s',guidance_scale) 
        log.info('strength: %s',strength) 
        log.info('init_image: %s',init_image) 
    
        if args.seed and i < 1:
            ran_seed = args.seed
        else:
            ran_seed = random.randrange(seed) #4294967294
        
        np.random.seed(int(ran_seed)) 
    
        log.info('Random Seed: %s',ran_seed)
        progress_callback = conn = None
        create_gif = False
        
        
        start_time = time.time()
        if use_fastsd:
            # hard coding height and width to 512 for now
            fastsd_h = 512
            fastsd_w = 512 
            output = engine(
                    prompt=prompt,
                    negative_prompt=None,
                    height=fastsd_h,
                    width=fastsd_w,
                    num_inference_steps=num_infer_steps,
                    guidance_scale=guidance_scale,
                    seed=ran_seed,
                )
        elif model_name == "sd_1.5_inpainting" or model_name == "sd_1.5_inpainting_int8":
            output = engine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_path=os.path.join(weight_path, "..", "cache1.png"),
                mask_path=os.path.join(weight_path, "..", "cache0.png"),
                scheduler=scheduler,
                strength=strength,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                callback=progress_callback,
                callback_userdata=conn
            )
        elif model_name == "controlnet_referenceonly":
            output = engine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=Image.open(init_image),
                scheduler=scheduler,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                eta=0.0,
                create_gif=bool(create_gif),
                model=model_path,
                callback=progress_callback,
                callback_userdata=conn
            )
        elif "controlnet" in model_name: 
            output = engine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=Image.open(init_image),
                scheduler=scheduler,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                eta=0.0,
                create_gif=bool(create_gif),
                model=model_path,
                callback=progress_callback,
                callback_userdata=conn
            )        
        elif model_name == "sd_1.5_square_lcm":        
            output = engine(
                 prompt=prompt,
                 negative_prompt=None,
                 num_inference_steps=num_infer_steps,
                 guidance_scale=guidance_scale,
                 seed=ran_seed,
                 callback=progress_callback,
                 callback_userdata=conn,
            )
        elif "sdxl" in model_name:        
            output = engine(
                 prompt=prompt,
                 negative_prompt=None,
                 num_inference_steps=num_infer_steps,
                 guidance_scale=guidance_scale,
                 seed=ran_seed,
                 callback=progress_callback,
                 callback_userdata=conn,
            )            
        elif "sd_3.0_med" in model_name or "sd_3.5_med" in model_name:
            if model_name =="sd_3.5_med_turbo_square":
                negative_prompt=None
            
            output = engine(
                 prompt=prompt,
                 negative_prompt=negative_prompt,
                 num_inference_steps=num_infer_steps,
                 guidance_scale=guidance_scale,
                 seed=ran_seed,
                 callback=progress_callback,
                 callback_userdata=conn,
            )                           
       
        else:
            if model_name == "sd_2.1_square":
                scheduler = EulerDiscreteScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    prediction_type="v_prediction"
                )
            model = model_path
            if "sd_2.1" in model_name:
                model = model_name

            output = engine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=None if init_image is None else Image.open(init_image),
                scheduler=scheduler,
                strength=strength,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                eta=0.0,
                create_gif=bool(create_gif),
                model=model,
                callback=progress_callback,
                callback_userdata=conn
            )


        gen_time = time.time() - start_time
        print (f"Image Generation Time: {round(gen_time,2)} seconds")
        if use_fastsd:
            model_name = model_name.replace("/","_")
        results.append([output,model_name 
                        + "_" + timestamp_24 
                        + "_" + '_'.join(map(str,execution_devices)) 
                        + "_" + str(ran_seed) 
                        + "_" + str(num_infer_steps) 
                        + "_steps",gen_time])
        
        generation_time.append(gen_time)

    if args.num_images > 1:
        print(f"Average Image Generation Time: {round(mean(generation_time),2)} seconds")

    if args.save_image:
        index = 1
        for result in results:
            if not use_fastsd and "sd_3.0" not in model_name and "lcm" not in model_name and "sdxl" not in model_name:
                cv2.imwrite(result[1] + "_" + str(index) + ".jpg", result[0])                         
            else:
                result[0].save(result[1] + "_" + str(index) + ".jpg")
            index += 1 
            

if __name__ == "__main__":
    sys.exit(main())

