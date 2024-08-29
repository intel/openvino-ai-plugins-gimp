"""
Copyright(C) 2022-2023 Intel Corporation
SPDX - License - Identifier: Apache - 2.0

"""
from tokenize import untokenize

import inspect
from typing import List, Optional, Union, Dict
import numpy as np
# openvino

# tokenizer
from transformers import CLIPTokenizer
import torch

from diffusers import DiffusionPipeline
from diffusers import UniPCMultistepScheduler,DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler
import cv2
import os
import sys


#For GIF
import PIL
from PIL import Image
import glob
import json
import time

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

from openvino.runtime import Model, Core
from collections import namedtuple

from controlnet_aux import HEDdetector
from typing import Union, List, Optional, Tuple


class HEDOVModel:
    """ Helper wrapper for HED model inference"""
    def __init__(self, core, model_path, device="AUTO"):
        self.core = core
        self. model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model, device)

    def __call__(self, input_tensor:torch.Tensor):
        """
        inference step
        
        Parameters:
          input_tensor (torch.Tensor): tensor with prerpcessed input image
        Returns:
           predicted keypoints heatmaps
        """
        h, w = input_tensor.shape[2:]
        input_shape = self.model.input(0).shape
        if h != input_shape[2] or w != input_shape[3]:
            self.reshape_model(h, w)
        results = self.compiled_model(input_tensor)
        return torch.from_numpy(results[self.compiled_model.output(0)]), torch.from_numpy(results[self.compiled_model.output(1)])

    def reshape_model(self, height:int, width:int):
        """
        helper method for reshaping model to fit input data
        
        Parameters:
          height (int): input tensor height
          width (int): input tensor width
        Returns:
          None
        """
        self.model.reshape({0: [1, 3, height, width]})
        self.compiled_model = self.core.compile_model(self.model)

    def parameters(self):
        Device = namedtuple("Device", ["device"])
        return [Device(torch.device("cpu"))]
        



def scale_fit_to_window(dst_width:int, dst_height:int, image_width:int, image_height:int):
    """
    Preprocessing helper function for calculating image size for resize with peserving original aspect ratio
    and fitting image to specific window size

    Parameters:
      dst_width (int): destination window width
      dst_height (int): destination window height
      image_width (int): source image width
      image_height (int): source image height
    Returns:
      result_width (int): calculated width for resize
      result_height (int): calculated height for resize
    """
    im_scale = min(dst_height / image_height, dst_width / image_width)
    return int(im_scale * image_width), int(im_scale * image_height)

def preprocess(image: PIL.Image.Image):
    """
    Image preprocessing function. Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
    then converts it to np.ndarray and adds padding with zeros on right or bottom side of image (depends from aspect ratio), after that
    converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally, converts data layout from planar NHWC to NCHW.
    The function returns preprocessed input tensor and padding size, which can be used in postprocessing.

    Parameters:
      image (PIL.Image.Image): input image
    Returns:
       image (np.ndarray): preprocessed image tensor
       meta (Dict): dictionary with preprocessing metadata info
    """

    src_width, src_height = image.size
    #image = image.convert('RGB')
    dst_width, dst_height = scale_fit_to_window(
        512, 512, src_width, src_height)
    image = np.array(image.resize((dst_width, dst_height),
                     resample=PIL.Image.Resampling.LANCZOS))[None, :]

    pad_width = 512 - dst_width
    pad_height = 512 - dst_height
    pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
    image = np.pad(image, pad, mode="constant")
    image = image.astype(np.float32) / 255.0
    #image = 2.0 * image - 1.0
    image = image.transpose(0, 3, 1, 2)

    return image, pad


    
    
def randn_tensor(
    shape: Union[Tuple, List],
    dtype: Optional[np.dtype] = np.float32,
):
    """
    Helper function for generation random values tensor with given shape and data type
    
    Parameters:
      shape (Union[Tuple, List]): shape for filling random values
      dtype (np.dtype, *optiona*, np.float32): data type for result
    Returns:
      latents (np.ndarray): tensor with random values with given data type and shape (usually represents noise in latent space)
    """
    latents = np.random.randn(*shape).astype(dtype)

    return latents

class ControlNetScribble(DiffusionPipeline):
    def __init__(
            self,
             #scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
            model="runwayml/stable-diffusion-v1-5",
            tokenizer="openai/clip-vit-large-patch14",
            device=["CPU","CPU","CPU"],
            ):
            
        super().__init__()    
            
        self.set_progress_bar_config(disable=False)    

        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model,local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
            self.tokenizer.save_pretrained(model)



        #scheduler =   UniPCMultistepScheduler.from_pretrained(os.path.join(model,"UniPCMultistepScheduler_config"))
        
    
        
        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')}) #adding caching to reduce init time
        print("Setting caching")
        
       
        HED_OV_PATH = os.path.join(model, "hed.xml")
        self.hed_estimator = HEDdetector.from_pretrained('lllyasviel/Annotators') 
        
    
        
        ov_hed = HEDOVModel(self.core, HED_OV_PATH, device="CPU")
        self.hed_estimator.netNetwork.model = ov_hed
        

        controlnet = os.path.join(model, "controlnet-scribble.xml")
        text_encoder = os.path.join(model, "text_encoder.xml")
        unet = os.path.join(model, "unet_controlnet.xml")

        vae_decoder = os.path.join(model, "vae_decoder.xml")

        ####################
        self.load_models(self.core, device, controlnet, text_encoder, unet, vae_decoder)
        

        # encoder
        self.vae_encoder = None
        self._vae_d_output = self.vae_decoder.output(0)
        self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder is not None else None
        
        self.height = self.unet.input(0).shape[2] * 8
        self.width = self.unet.input(0).shape[3] * 8    

 
    def load_models(self, core: Core, device: str, controlnet:Model, text_encoder: Model, unet: Model, vae_decoder: Model):
        """
        Function for loading models on device using OpenVINO
        
        Parameters:
          core (Core): OpenVINO runtime Core class instance
          device (str): inference device
          controlnet (Model): OpenVINO Model object represents ControlNet
          text_encoder (Model): OpenVINO Model object represents text encoder
          unet (Model): OpenVINO Model object represents UNet
          vae_decoder (Model): OpenVINO Model object represents vae decoder
        Returns
          None
        """
        start = time.time()
        self.text_encoder = core.compile_model(text_encoder, device[0])
        self.text_encoder_out = self.text_encoder.output(0)
        print("text encoder loaded in:", time.time() - start)
        start = time.time()
        self.controlnet = core.compile_model(controlnet, device[2])
        print("controlnet loaded in:", time.time() - start)
        start = time.time()
        self.unet = core.compile_model(unet, device[1])
        self.unet_out = self.unet.output(0)
        #self.unet_neg = core.compile_model(unet_neg, device[2])
        #self.unet_neg_out = self.unet_neg.output(0)
        print("unet loaded in:", time.time() - start)
        start = time.time()
        self.vae_decoder = core.compile_model(vae_decoder, device[2])
        self.vae_decoder_out = self.vae_decoder.output(0)
        print("vae decoder loaded in:", time.time() - start)
        
      
        
        

    def __call__(
            self,
            prompt,
            image: Image.Image=None,
            negative_prompt=None,
            num_inference_steps = 32,
            guidance_scale = 7.5,
            controlnet_conditioning_scale: float = 1.0,
            eta = 0.0,
            create_gif = False,
            model = None,
            callback = None,
            callback_userdata = None,
            do_hed = True,
            scheduler=None
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        # 2. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, negative_prompt=negative_prompt)

        
        # 3. Preprocess image
        image = image.convert("RGB")
        if do_hed :
            hed = self.hed_estimator(image)
        else:
            hed = image
    
        orig_width, orig_height = hed.size
        
        hed, pad = preprocess(hed)
        
          
        height, width = hed.shape[-2:]
        if do_classifier_free_guidance:
            hed = np.concatenate(([hed] * 2))
        
        
        # 4. set timesteps
 
        
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        


        # 6. Prepare latent variables
        num_channels_latents = 4
        batch_size = 1
        #timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, scheduler)
        #latent_timestep = timesteps[:1]

        # get the initial random noise unless the user supplied it
        
        latents = self.prepare_latents(batch_size,num_channels_latents,height,width,scheduler)


        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
    
        if create_gif:
            frames = []


        # 7. Denoising loop
        # num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        #    for i, t in enumerate(timesteps):
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

        #for i, t in enumerate(self.progress_bar(timesteps)):
                if callback:
                   callback(i, callback_userdata)

            # expand the latents if we are doing classifier free guidance
            #noise_pred = []
                latent_model_input = np.concatenate(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                #print("latent_model_input", latent_model_input)
                
             
                
                result = self.controlnet([latent_model_input, t, text_embeddings, hed])
                #print("result", result)
                down_and_mid_blok_samples = [sample * controlnet_conditioning_scale for _, sample in result.items()]
                
                # predict the noise residual
                noise_pred = self.unet([latent_model_input, t, text_embeddings, *down_and_mid_blok_samples])[self.unet_out]
                #print("noise_pred:", noise_pred)


                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()
                #print("latents", latents)

                if create_gif:
                    frames.append(latents)
                    
                # update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()                    

        if callback:
              callback(num_inference_steps, callback_userdata)

        # decode_latents
        # scale and decode the image latents with vae

        # 8. Post-processing
        image = self.decode_latents(latents, pad)  
        output_type = "pil"
        #print("output_type",output_type)
     
        
        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            image = [img.resize((orig_width, orig_height), Image.Resampling.LANCZOS) for img in image]
            
        else:
            image = [cv2.resize(img, (orig_width, orig_width))
                     for img in image]
            

        if create_gif:
            gif_folder=os.path.join(model,"../../../gif")
            print("gif_folder:",gif_folder)


        return image[0]
        
    def _encode_prompt(self, prompt:Union[str, List[str]], num_images_per_prompt:int = 1, do_classifier_free_guidance:bool = True, negative_prompt:Union[str, List[str]] = None):
        """
        Encodes the prompt into text encoder hidden states.

        Parameters:
            prompt (str or list(str)): prompt to be encoded
            num_images_per_prompt (int): number of images that should be generated per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(
            text_input_ids)[self.text_encoder_out]

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(
                text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(
                text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self.text_encoder_out]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings        
        
    def decode_latents(self, latents:np.array, pad:Tuple[int]):
        """
        Decode predicted image from latent space using VAE Decoder and unpad image result
        
        Parameters:
           latents (np.ndarray): image encoded in diffusion latent space
           pad (Tuple[int]): each side padding sizes obtained on preprocessing step
        Returns:
           image: decoded by VAE decoder image
        """
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latents)[self.vae_decoder_out]
        #print("Decode_image shape", image.shape)
        (_, end_h), (_, end_w) = pad[1:3]
        h, w = image.shape[2:]
        unpad_h = h - end_h
        unpad_w = w - end_w
        image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        #print("Inside decode", image.shape)
        return image    

    def prepare_latents(self,batch_size,num_channels_latents,height, width,scheduler):
        """
        Preparing noise to image generation. If initial latents are not provided, they will be generated randomly, 
        then prepared latents scaled by the standard deviation required by the scheduler
        
        Parameters:
           batch_size (int): input batch size
           num_channels_latents (int): number of channels for noise generation
      
        Returns:
           latents (np.ndarray): scaled initial noise for diffusion
        """
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
       
        latents = randn_tensor(shape, np.float32)
       

        # scale the initial noise by the standard deviation required by the scheduler
        if isinstance(scheduler, LMSDiscreteScheduler):
            
            latents = latents * scheduler.sigmas[0].numpy()
        elif isinstance(scheduler, EulerDiscreteScheduler):
            
            latents = latents * scheduler.sigmas.max().numpy()
        else:
            latents = latents * scheduler.init_noise_sigma

        #latents = latents * self.scheduler.init_noise_sigma.numpy()
        return latents
        

class ControlNetScribbleAdvanced(DiffusionPipeline):
    def __init__(
            self,
            model="runwayml/stable-diffusion-v1-5",
            tokenizer="openai/clip-vit-large-patch14",
            device=["CPU","CPU","CPU","CPU"],
            blobs=False,
            swap=False
            ):

        super().__init__()
        self.vae_scale_factor = 8
        self.set_progress_bar_config(disable=False)
        
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model,local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
            self.tokenizer.save_pretrained(model)

        self.swap = swap
   
        
        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')}) #adding caching to reduce init time
        print("Setting caching")

        HED_OV_PATH = os.path.join(model, "hed.xml")
        self.hed_estimator = HEDdetector.from_pretrained('lllyasviel/Annotators') 
        
    
        
        ov_hed = HEDOVModel(self.core, HED_OV_PATH, device="CPU")
        self.hed_estimator.netNetwork.model = ov_hed
        



   
        controlnet = os.path.join(model, "controlnet-scribble.xml")
     
      
        text_encoder = os.path.join(model, "text_encoder.xml")
        unet_int8_model = os.path.join(model, "unet_controlnet_int8.xml")
        unet_time_proj_model = os.path.join(model, "unet_time_proj_sym.xml")
        vae_decoder = os.path.join(model, "vae_decoder.xml")
        
        #self.npu_flag = False
        #self.npu_flag_neg = False

        ####################
        self.load_models(self.core, device, controlnet, text_encoder, unet_time_proj_model, unet_int8_model, vae_decoder, blobs, model)
        # self.set_progress_bar_config(disable=True)

        # encoder
        self.vae_encoder = None
        self._vae_d_output = self.vae_decoder.output(0)
        self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder is not None else None
        
        self.height = self.unet.input(0).shape[2] * 8
        self.width = self.unet.input(0).shape[3] * 8  
        print("All models loaded")
        
        print("create infer request")

        self.infer_request_neg = self.unet_neg.create_infer_request()
        self.infer_request = self.unet.create_infer_request()
        self.infer_request_time_proj = self.unet_time_proj.create_infer_request()
        self.infer_request_controlnet = self.controlnet.create_infer_request()
        print("create infer request created")        
        


    def load_models(self, core: Core, device: str, controlnet:Model, text_encoder: Model, unet_time_proj_model:Model, unet_int8_model: Model, vae_decoder: Model, blobs: bool, model: str):
        """
        Function for loading models on device using OpenVINO
        
        Parameters:
          core (Core): OpenVINO runtime Core class instance
          device (str): inference device
          controlnet (Model): OpenVINO Model object represents ControlNet
          text_encoder (Model): OpenVINO Model object represents text encoder
          unet (Model): OpenVINO Model object represents UNet
          vae_decoder (Model): OpenVINO Model object represents vae decoder
        Returns
          None
        """
        start = time.time()
        self.text_encoder = core.compile_model(text_encoder, device[0])
        self.text_encoder_out = self.text_encoder.output(0)
        print("text encoder loaded in:", time.time() - start)
        start = time.time()
        
        self.controlnet = core.compile_model(controlnet, "GPU")
        print("controlnet loaded in:", time.time() - start)
        start = time.time()
        
        print(" compile unet_time_proj")
        self.unet_time_proj = core.compile_model(unet_time_proj_model, "CPU")        
        
        if blobs:
            blob_name = "unet_controlnet_int8.blob" 
            if "NPU" in device[1]:      
                print("Loading unet blob on npu:",blob_name)
                start = time.time()
                with open(os.path.join(model, blob_name), "rb") as f:
                    self.unet = self.core.import_model(f.read(), device[1])
                print("unet loaded on npu in:", time.time() - start)
                self.npu_flag = True
            
            else:
                print("compiling start on ",device[1])
                start = time.time()
                self.unet = self.core.compile_model(os.path.join(model, unet_int8_model), device[1])
                print("compiling done in ", time.time() - start)
                self.npu_flag = False

            # Negative Prompt
            if device[1] == device[2]:
                self.unet_neg = self.unet
                self.npu_flag_neg = self.npu_flag

            else:
                if "NPU" in device[2]:   
                    print("Loading unet blob on npu:",blob_name) 
                    start = time.time()
                    with open(os.path.join(model, blob_name), "rb") as f:
                        self.unet_neg = self.core.import_model(f.read(), device[2])
                    print("unet loaded on npu in:", time.time() - start)
                    self.npu_flag_neg = True                        
                else:
                    print("compiling start on ",device[1])
                    start = time.time()              
                    self.unet_neg = self.core.compile_model(os.path.join(model, unet_int8_model), device[2])  
                    print("compiling done in ", time.time() - start)
                    self.npu_flag_neg = False
    
        else:
            self.npu_flag_neg = False
            self.npu_flag = False
            self.unet = self.core.compile_model(os.path.join(model, unet_int8_model), device[1])
            self.unet_neg = self.core.compile_model(os.path.join(model, unet_int8_model), device[2])

        start = time.time()
        self.vae_decoder = core.compile_model(vae_decoder, device[3])
        self.vae_decoder_out = self.vae_decoder.output(0)
        print("vae decoder loaded in:", time.time() - start)
        
      

        

    def __call__(
            self,
            prompt,
            image: Image.Image=None,
            negative_prompt=None,
            scheduler=None,
            num_inference_steps = 32,
            guidance_scale = 7.5,
            controlnet_conditioning_scale: float = 1.0,
            eta = 0.0,
            create_gif = False,
            model = None,
            callback = None,
            callback_userdata = None,
            do_hed = True
            #scheduler=None,
    ):
        
        
        do_classifier_free_guidance = guidance_scale > 1.0
        # 2. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, negative_prompt=negative_prompt)

        
        # 3. Preprocess image
        image = image.convert("RGB")
        if do_hed :
            hed = self.hed_estimator(image)
        else:
            hed = image
    
        orig_width, orig_height = hed.size
        
        hed, pad = preprocess(hed)
        
          
        height, width = hed.shape[-2:]
        if do_classifier_free_guidance:
            hed = np.concatenate(([hed] * 2))
          
        
        
        # 4. set timesteps
        # set timesteps
        
        #print("scheduler",scheduler)
        
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        


        # 6. Prepare latent variables
        num_channels_latents = 4
        batch_size = 1
        #timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, scheduler)
        #latent_timestep = timesteps[:1]

        # get the initial random noise unless the user supplied it
        
        latents = self.prepare_latents(batch_size,num_channels_latents,height,width,scheduler)


        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
    
        if create_gif:
            frames = []
        

            

        # 7. Denoising loop

        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

    
                if callback:
                   callback(i, callback_userdata)

                noise_pred = []
                latent_model_input = latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                
                latent_model_input_2 = np.concatenate(
                    [latents] * 2) if do_classifier_free_guidance else latents    
                    
                latent_model_input_2 = scheduler.scale_model_input(latent_model_input_2, t)


       
                #result = self.controlnet([latent_model_input_2, t, text_embeddings, pose])  
                controlnet_dict = {"sample":latent_model_input_2, "timestep":t, "encoder_hidden_states":text_embeddings, "controlnet_cond":hed}
                result = self.infer_request_controlnet.infer(controlnet_dict, share_outputs = True)

                tensor_dict_neg = {}
                tensor_dict = {}

                for k,v in result.items():
                    tensor_name = next(iter(k.names))
                    #print("tensor_name--", tensor_name)
                    vneg = np.expand_dims(v[0], axis=0)
                    tensor_dict_neg[tensor_name] = vneg #controlnet_conditioning_scale * vneg #.astype(np.float32)
                    
                    vpos = np.expand_dims(v[1], axis=0)
                    tensor_dict[tensor_name] = vpos #controlnet_conditioning_scale * vpos #.astype(np.float32)                  
      
                
            
                    
                time_proj_dict = {"timestep" : t}
                self.infer_request_time_proj.start_async(time_proj_dict,share_inputs = True)
                self.infer_request_time_proj.wait()
                time_proj = self.infer_request_time_proj.get_output_tensor(0).data.astype(np.float32) 
                
                ##### NEGATIVE PIPELINE #####
                input_dict_neg = {"sample":latent_model_input, "time_proj": time_proj, "encoder_hidden_states":np.expand_dims(text_embeddings[0], axis=0)}
                input_dict_neg.update(tensor_dict_neg)
                
                if self.npu_flag_neg:
                    input_dict_neg_final = {k: v for k, v in sorted(input_dict_neg.items(), key=lambda x: x[0])}
                else:
                    input_dict_neg_final = input_dict_neg
                
                self.infer_request_neg.start_async(input_dict_neg_final, share_inputs = True)
                
                
                ##### POSITIVE PIPELINE #####
                input_dict = {"sample":latent_model_input, "time_proj": time_proj, "encoder_hidden_states":np.expand_dims(text_embeddings[1], axis=0)}
                input_dict.update(tensor_dict)
                if self.npu_flag:
                    input_dict_final = {k: v for k, v in sorted(input_dict.items(), key=lambda x: x[0])}
                else:
                    input_dict_final = input_dict

                self.infer_request.start_async(input_dict_final,share_inputs = True)
                self.infer_request_neg.wait()
                self.infer_request.wait()                    
                
                noise_pred_neg = self.infer_request_neg.get_output_tensor(0)
                noise_pred_pos = self.infer_request.get_output_tensor(0) 

                noise_pred.append(noise_pred_neg.data.astype(np.float32))
                noise_pred.append(noise_pred_pos.data.astype(np.float32))  

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)   
                    
                    
                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()
                #print("latents", latents)

                if create_gif:
                    frames.append(latents)
                # update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()                    

        if callback:
              callback(num_inference_steps, callback_userdata)

        # 8. Post-processing
        image = self.decode_latents(latents, pad)  
        output_type = "pil"
   
        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            image = [img.resize((orig_width, orig_height), Image.Resampling.LANCZOS) for img in image]
         
        else:
            image = [cv2.resize(img, (orig_width, orig_width))
                     for img in image]
         


             

        if create_gif:
            gif_folder=os.path.join(model,"../../../gif")
            print("gif_folder:",gif_folder)
            if not os.path.exists(gif_folder):
                os.makedirs(gif_folder)
            for i in range(0,len(frames)):
                image = self.decode_latents(frames[i], pad)  
                image = self.numpy_to_pil(image)
                image = [img.resize((orig_width, orig_height), Image.Resampling.LANCZOS) for img in image]                
                output = gif_folder + "/" + str(i).zfill(3) +".png"
                image[0].save(output)
         
            with open(os.path.join(gif_folder, "prompt.json"), "w") as file:
                json.dump({"prompt": prompt}, file)
            frames_image =  [Image.open(image) for image in glob.glob(f"{gif_folder}/*.png")]
            frame_one = frames_image[0]
            gif_file=os.path.join(gif_folder,"stable_diffusion.gif")
            frame_one.save(gif_file, format="GIF", append_images=frames_image, save_all=True, duration=100, loop=0)


        return image[0]
        
    def _encode_prompt(self, prompt:Union[str, List[str]], num_images_per_prompt:int = 1, do_classifier_free_guidance:bool = True, negative_prompt:Union[str, List[str]] = None):
        """
        Encodes the prompt into text encoder hidden states.

        Parameters:
            prompt (str or list(str)): prompt to be encoded
            num_images_per_prompt (int): number of images that should be generated per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(
            text_input_ids)[self.text_encoder_out]

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(
                text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(
                text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self.text_encoder_out]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings        
        
    def decode_latents(self, latents:np.array, pad:Tuple[int]):
        """
        Decode predicted image from latent space using VAE Decoder and unpad image result
        
        Parameters:
           latents (np.ndarray): image encoded in diffusion latent space
           pad (Tuple[int]): each side padding sizes obtained on preprocessing step
        Returns:
           image: decoded by VAE decoder image
        """
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latents)[self.vae_decoder_out]
        #print("Decode_image shape", image.shape)
        (_, end_h), (_, end_w) = pad[1:3]
        h, w = image.shape[2:]
        unpad_h = h - end_h
        unpad_w = w - end_w
        image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        #print("Inside decode", image.shape)
        return image    

    def prepare_latents(self,batch_size,num_channels_latents,height, width,scheduler): #, scheduler):
        """
        Preparing noise to image generation. If initial latents are not provided, they will be generated randomly, 
        then prepared latents scaled by the standard deviation required by the scheduler
        
        Parameters:
           batch_size (int): input batch size
           num_channels_latents (int): number of channels for noise generation
      
        Returns:
           latents (np.ndarray): scaled initial noise for diffusion
        """
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
       
        latents = randn_tensor(shape, np.float32)
 
        # scale the initial noise by the standard deviation required by the scheduler
        if isinstance(scheduler, LMSDiscreteScheduler):
            
            latents = latents * scheduler.sigmas[0].numpy()
        elif isinstance(scheduler, EulerDiscreteScheduler):
            
            latents = latents * scheduler.sigmas.max().numpy()
        else:
            latents = latents * scheduler.init_noise_sigma

        #latents = latents * scheduler.init_noise_sigma.numpy()
        return latents


if __name__ == "__main__":
    weight_path = os.path.join(os.path.expanduser('~'), "openvino-ai-plugins-gimp", "weights")
    
    model_path = os.path.join(weight_path, "stable-diffusion-ov/controlnet-scribble")
    device_name = ["GPU.1", "GPU.1" , "GPU.1"]
    
    prompt = "Dancing Darth Vader, best quality, extremely detailed"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    seed = 42
    num_infer_steps = 20
    guidance_scale = 7.5
    init_image = os.path.join(os.path.expanduser('~'),"Downloads","224540208-c172c92a-9714-4a7b-857a-b1e54b4d4791.jpg")
    

    if seed is not None:   
        np.random.seed(int(seed))
    else:
        ran_seed = random.randrange(4294967294) #4294967294 
        np.random.seed(int(ran_seed))
       
   
    engine = ControlNetScribble(
        model = model_path,
        device = device_name
    )
      
    output = engine(
    prompt = prompt,
    negative_prompt = negative_prompt,
    image = Image.open(init_image),

    num_inference_steps = num_infer_steps,
    guidance_scale = guidance_scale,
    eta = 0.0,
    create_gif = False,
    model = model_path,
    callback = None,
    callback_userdata = None
)
    

    
