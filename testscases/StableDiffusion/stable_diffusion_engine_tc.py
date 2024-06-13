import logging
import time
import sys
import random
import cv2
import argparse
import os
import json
import numpy as np

from PIL import Image
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, LCMScheduler, EulerDiscreteScheduler
plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","..","gimpopenvino","tools","openvino_common")
sys.path.extend([plugin_loc])


from models_ov.stable_diffusion_engine import StableDiffusionEngineAdvanced, StableDiffusionEngine, LatentConsistencyEngine, StableDiffusionEngineReferenceOnly

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    default_sd_model_path = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp", "weights","stable-diffusion-ov","stable-diffusion-1.5","square_int8")
    # fmt: off
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model_path',type = str, default = default_sd_model_path, required = False,
                      help='Optional. Modle path of directory. Default is ./sd-1.5-square-quantized_LNL/square_int8.')
    args.add_argument('-td','--text_device',type = str, default = 'GPU', required = False,
                      help='Optional. Specify the target device to infer on; CPU, GPU, NPU '
                      'is acceptable for Text encoder. Default value is GPU.')
    args.add_argument('-ud','--unet_device',type = str, default = 'GPU', required = False,
                      help='Optional. Specify the target device to infer on; CPU, GPU, NPU '
                      'is acceptable for Unet. Default value is GPU.')
    args.add_argument('-und','--unet_neg_device',type = str, default = 'NPU', required = False,
                      help='Optional. Specify the target device to infer on; CPU, GPU, NPU '
                      'is acceptable for Unet Negative. Default value is NPU.')
    args.add_argument('-vd','--vae_device',type = str, default = 'GPU', required = False,
                      help='Optional. Specify the target device to infer on; CPU, GPU, NPU '
                      'is acceptable for VAE decoder and encoder. Default value is GPU.')
    args.add_argument('-seed','--seed',type = int, default = None, required = False,
                      help='Optional. Specify the seed for initialize latent space.')
    args.add_argument('-niter','--iterations',type = int, default = 20, required = False,
                      help='Optional. Iterations for Stable diffusion.')
    args.add_argument('-si','--save_image',action='store_true', help='Optional. Save output image.')
    args.add_argument('-n','--num_images',type = int, default = 1, required = False,
                      help='Optional. Number of images to generate.')
    args.add_argument('-pm','--power_mode',type = str, default = None, required = False,
                      help='Optional. Specify the power mode.')
    

    
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = args.model_path
    results = []

    model_config_file_name = os.path.join(model_path, "config.json")
    
    try:
        if args.power_mode is not None and os.path.exists(model_config_file_name):
            with open(model_config_file_name, 'r') as file:
                model_config = json.load(file)
                if model_config['power modes supported'].lower() == "yes":
                    execution_devices = model_config[args.power_mode.lower()]
                else:
                    execution_devices = model_config['best performance']
        else:
            execution_devices = [args.text_device, args.unet_device, args.unet_neg_device, args.vae_device]
        

    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        log.error(f"Error loading configuration: {e}. Only CPU will be used.")


    log.info('Initializing Inference Engine...') 
    log.info('Model Path: %s',model_path ) 
    log.info('Run models on: %s',execution_devices) 
    
    engine_adv = StableDiffusionEngineAdvanced(
        model = model_path, 
        device = execution_devices, 
    )

    prompt = "a beautiful artwork illustration, concept art sketch of an astronaut in white futuristic cybernetic armor in a dark cave, volumetric fog, godrays, high contrast, vibrant colors, vivid colors, high saturation, by Greg Rutkowski and Jesper Ejsing and Raymond Swanland and alena aenami, featured on artstation, wide angle, vertical orientation" 
    negative_prompt = "lowres, bad quality, monochrome, cropped head, deformed face, bad anatomy" 
    
    init_image = None 
    num_infer_steps = args.iterations 
    guidance_scale = 8.0 
    strength = 0.8 
    seed = 4294967294   
    
    scheduler = EulerDiscreteScheduler( 
                    beta_start=0.00085,  
                    beta_end=0.012,  
                    beta_schedule="scaled_linear" 
    ) 
    

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

        start_time = time.time()
        image = engine_adv( 
            prompt = prompt, 
            negative_prompt = negative_prompt, 
            init_image = None if init_image is None else Image.open(init_image),  
            scheduler = scheduler, 
            strength = 1.0 if init_image is None else strength, 
            num_inference_steps = num_infer_steps, 
            guidance_scale = guidance_scale,
        )
        print ("Process time: ", time.time() - start_time)

        results.append([image,"sd_result" + 
            "_" + execution_devices[0] + 
            "_" + execution_devices[1] + 
            "_" + execution_devices[2] + 
            "_" + execution_devices[3] + 
            "_" + str(ran_seed) + 
            "_" + str(num_infer_steps) +  "_steps" +".jpg"])
        
    if args.save_image:
        for result in results:
            cv2.imwrite(result[1], result[0]) 
    
if __name__ == "__main__":
    sys.exit(main())

