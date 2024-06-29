import logging
import time
import sys
import random
import cv2
import argparse
import os
import json
import numpy as np
from statistics import mean
from gimpopenvino.tools.tools_utils import get_weight_path
from PIL import Image
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, LCMScheduler, EulerDiscreteScheduler
plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","..","gimpopenvino","tools","openvino_common")
sys.path.extend([plugin_loc])

from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, LCMScheduler, EulerDiscreteScheduler
from models_ov.stable_diffusion_engine import StableDiffusionEngineAdvanced, StableDiffusionEngine, LatentConsistencyEngine, StableDiffusionEngineReferenceOnly
from models_ov.stable_diffusion_engine_inpainting import StableDiffusionEngineInpainting
from models_ov.stable_diffusion_engine_inpainting_advanced import StableDiffusionEngineInpaintingAdvanced
from models_ov.stable_diffusion_3 import StableDiffusionThreeEngine
from models_ov.controlnet_openpose import ControlNetOpenPose
from models_ov.controlnet_canny_edge import ControlNetCannyEdge
from models_ov.controlnet_scribble import ControlNetScribble, ControlNetScribbleAdvanced
from models_ov.controlnet_openpose_advanced import ControlNetOpenPoseAdvanced
from models_ov.controlnet_cannyedge_advanced import ControlNetCannyEdgeAdvanced

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
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
    args.add_argument('-seed','--seed',type = int, default = None, required = False,
                      help='Optional. Specify the seed for initialize latent space.')
    args.add_argument('-niter','--iterations',type = int, default = 20, required = False,
                      help='Optional. Iterations for Stable diffusion.')
    # save output image
    args.add_argument('-si','--save_image',action='store_true', help='Optional. Save output image.')
    
    # generate multiple images
    args.add_argument('-n','--num_images',type = int, default = 1, required = False,
                      help='Optional. Number of images to generate.')
    # power mode
    args.add_argument('-pm','--power_mode',type = str, default = "best performance", required = False,
                      help='Optional. Specify the power mode. Default is best performance')
    # prompt, negative prompt
    args.add_argument('-pp','--prompt',type = str, default = "a bowl of cherries", required = False,
                      help='Optional. Specify the prompt.  Default: "a bowl of cherries"')
    args.add_argument('-np','--neg_prompt',type = str, default = "low quality, bad, low resolution, monochrome", required = False,
                      help='Optional. Specify the negative prompt.  Default: "low  quality, bad, low resolution, monochrome"')
         
    return parser.parse_args()


def initialize_engine(model_name, model_path, device_list):
    if model_name == "sd_1.5_square_int8":
        log.info('Device list: %s', device_list)
        return StableDiffusionEngineAdvanced(model=model_path, device=device_list)
    if model_name == "sd_3.0_square_int8" or model_name == "sd_3.0_square_int4":
        log.info('Device list: %s', device_list)
        return StableDiffusionThreeEngine(model=model_path, device=device_list)
    if model_name == "sd_1.5_inpainting":
        return StableDiffusionEngineInpainting(model=model_path, device=device_list)
    if model_name == "sd_1.5_square_lcm":
        return LatentConsistencyEngine(model=model_path, device=device_list)
    if model_name == "sd_1.5_inpainting_int8":
        log.info('Advanced Inpainting Device list: %s', device_list)
        return StableDiffusionEngineInpaintingAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_openpose_int8":
        log.info('Device list: %s', device_list)
        return ControlNetOpenPoseAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_canny_int8":
        log.info('Device list: %s', device_list)
        return ControlNetCannyEdgeAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_scribble_int8":
        log.info('Device list: %s', device_list)
        return ControlNetScribbleAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_canny":
        return ControlNetCannyEdge(model=model_path, device=device_list)
    if model_name == "controlnet_scribble":
        return ControlNetScribble(model=model_path, device=device_list)
    if model_name == "controlnet_openpose":
        return ControlNetOpenPose(model=model_path, device=device_list)
    if model_name == "controlnet_referenceonly":
        return StableDiffusionEngineReferenceOnly(model=model_path, device=device_list)
    return StableDiffusionEngine(model=model_path, device=device_list)


def main():
    args = parse_args()
    results = []
    generation_time = []

    if args.model_base_path:
        weight_path = args.model_base_path
    else:
        weight_path = get_weight_path()
    
    # Check if the directory path exists
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"The directory path {weight_path} does not exist.")
    
    execution_devices = ["GPU"]*4
    
    model_paths = {
        "sd_1.4": ["stable-diffusion-ov", "stable-diffusion-1.4"],
        "sd_1.5_square_lcm": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_lcm"],
        "sd_1.5_portrait": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait"],
        "sd_1.5_square": ["stable-diffusion-ov", "stable-diffusion-1.5", "square"],
        "sd_1.5_square_int8": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_int8"],
        "sd_1.5_landscape": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape"],
        "sd_1.5_portrait_512x768": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait_512x768"],
        "sd_1.5_landscape_768x512": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape_768x512"],
        "sd_1.5_inpainting": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting"],
        "sd_1.5_inpainting_int8": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting_int8"],
        "sd_2.1_square_base": ["stable-diffusion-ov", "stable-diffusion-2.1", "square_base"],
        "sd_2.1_square": ["stable-diffusion-ov", "stable-diffusion-2.1", "square"],
        "sd_3.0_square_int8": ["stable-diffusion-ov", "stable-diffusion-3.0", "square_int8"],
        "sd_3.0_square_int4": ["stable-diffusion-ov", "stable-diffusion-3.0", "square_int4"],
        "controlnet_referenceonly": ["stable-diffusion-ov", "controlnet-referenceonly"],
        "controlnet_openpose": ["stable-diffusion-ov", "controlnet-openpose"],
        "controlnet_canny": ["stable-diffusion-ov", "controlnet-canny"],
        "controlnet_scribble": ["stable-diffusion-ov", "controlnet-scribble"],
        "controlnet_openpose_int8": ["stable-diffusion-ov", "controlnet-openpose-int8"],
        "controlnet_canny_int8": ["stable-diffusion-ov", "controlnet-canny-int8"],
        "controlnet_scribble_int8": ["stable-diffusion-ov", "controlnet-scribble-int8"],
    }
    model_name = args.model_name
    model_path = os.path.join(weight_path, *model_paths.get(model_name))    
    model_config_file_name = os.path.join(model_path, "config.json")

    try:
        if args.power_mode is not None and os.path.exists(model_config_file_name):
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

    log.info('Initializing Inference Engine...') 
    log.info('Model Path: %s',model_path ) 
    log.info('Run models on: %s',execution_devices) 

    prompt = args.prompt #"a beautiful artwork illustration, concept art sketch of an astronaut in white futuristic cybernetic armor in a dark cave, volumetric fog, godrays, high contrast, vibrant colors, vivid colors, high saturation, by Greg Rutkowski and Jesper Ejsing and Raymond Swanland and alena aenami, featured on artstation, wide angle, vertical orientation" 
    negative_prompt = args.neg_prompt # "lowres, bad quality, monochrome, cropped head, deformed face, bad anatomy" 
    
    init_image = None 
    num_infer_steps = args.iterations 
    guidance_scale = 8.0 
    strength = 1.0
    seed = 4294967294   
    
    scheduler = EulerDiscreteScheduler( 
                    beta_start=0.00085,  
                    beta_end=0.012,  
                    beta_schedule="scaled_linear" 
    ) 
    
    engine = initialize_engine(model_name=model_name, model_path=model_path, device_list=execution_devices)


    for i in range(0,args.num_images):
        log.info('Starting inference...') 
        log.info('Prompt: %s',prompt) 
        log.info('negative_prompt: %s',negative_prompt) 
        log.info('num_inference_steps: %s',num_infer_steps) 
        log.info('guidance_scale: %s',guidance_scale) 
        log.info('strength: %s',strength) 
        log.info('init_image: %s',init_image) 
    
        if args.seed:
            ran_seed = args.seed
        else:
            ran_seed = random.randrange(seed) #4294967294
        
        np.random.seed(int(ran_seed)) 
    
        log.info('Random Seed: %s',ran_seed)
        progress_callback = conn = None
        create_gif = False
        
        start_time = time.time()
        
        if model_name == "sd_1.5_inpainting" or model_name == "sd_1.5_inpainting_int8":
            output = engine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=Image.open(os.path.join(weight_path, "..", "cache1.png")),
                mask_image=Image.open(os.path.join(weight_path, "..", "cache0.png")),
                scheduler=scheduler,
                strength=strength,
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
            scheduler = LCMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear"
            )
            output = engine(
                prompt=prompt,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                scheduler=scheduler,
                lcm_origin_steps=50,
                model=model_path,
                callback=progress_callback,
                callback_userdata=conn,
                seed=ran_seed
            )
        elif "sd_3.0" in model_name:
            output = engine(
                    prompt = prompt,
                    negative_prompt = negative_prompt,
                    num_inference_steps = num_infer_steps,
                    guidance_scale = guidance_scale,
                    callback = progress_callback,
                    callback_userdata = conn,
                    seed = ran_seed
            )        
        else: # Covers SD 1.5 Square, Square INT8, SD 2.0
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
        results.append([output,model_name + "_" 
                        + '_'.join(map(str,execution_devices)) 
                        + "_" + str(ran_seed) 
                        + "_" + str(num_infer_steps) 
                        + "_steps",gen_time])
        
        generation_time.append(gen_time)

    if args.num_images > 1:
        print(f"Average Image Generation Time: {round(mean(generation_time),2)} seconds")

    if args.save_image:
        index = 1
        for result in results:
            if "sd_3.0" not in model_name and "lcm" not in model_name:
                cv2.imwrite(result[1] + "_" + str(index) + ".jpg", result[0])                         
            else:
                result[0].save(result[1] + "_" + str(index) + ".jpg")
            index += 1 
            

if __name__ == "__main__":
    sys.exit(main())

