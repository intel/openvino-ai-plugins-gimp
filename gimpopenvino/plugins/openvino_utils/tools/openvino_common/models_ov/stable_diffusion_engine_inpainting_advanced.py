"""
Copyright(C) 2022-2023 Intel Corporation
SPDX - License - Identifier: Apache - 2.0

"""
import inspect
from typing import List, Optional, Union, Dict
import numpy as np
# openvino
from openvino.runtime import Core, Model
# tokenizer
from transformers import CLIPTokenizer
import torch

from diffusers import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler
import cv2
import os
import sys


#For GIF
import PIL
from PIL import Image
import glob
import json

import time

def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)       
        
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
    
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
     
        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)
        

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)
    print("image shape", image.shape)
    print("mask shape", mask.shape)
    print("masked_image shape", masked_image.shape)
  

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image 

def result(var):
    return next(iter(var.values()))


class StableDiffusionEngineInpaintingAdvanced(DiffusionPipeline):
    def __init__(
            self,
            #scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
            model="runwayml/stable-diffusion-inpainting",
            tokenizer="openai/clip-vit-large-patch14",
            device=["CPU","CPU","CPU","CPU"],
            blobs=False
            
            ):
            
        #self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        try: 
            self.tokenizer = CLIPTokenizer.from_pretrained(model,local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
            self.tokenizer.save_pretrained(model)

        # models
        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')})  # Adding caching to reduce init time
        print("Setting caching")
        
        print("Text Device:", device[0])
        self.text_encoder = self.load_model(model, "text_encoder", device[0])
        self._text_encoder_output = self.text_encoder.output(0)

        print("unet Device:", device[1])
        print("unet-neg Device:", device[2])
        self.unet_time_proj = self.core.compile_model(os.path.join(model, "unet_time_proj.xml"), 'CPU')

        self.unet = self.load_model(model, "unet_int8", device[1])
        self.unet_neg = self.unet if device[1] == device[2] else self.load_model(model, "unet_int8", device[2])

        print("VAE Device:", device[3])
        self.vae_decoder = self.load_model(model, "vae_decoder", device[3])
        self.vae_encoder = self.load_model(model, "vae_encoder", device[3])

        self._vae_d_output = self.vae_decoder.output(0)
        self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder is not None else None
        self.height = self.unet.input("latent_model_input").shape[2] * 8
        self.width = self.unet.input("latent_model_input").shape[3] * 8

        self.infer_request_neg = self.unet_neg.create_infer_request()
        self.infer_request = self.unet.create_infer_request()
        self.infer_request_time_proj = self.unet_time_proj.create_infer_request()


    def load_model(self, model, model_name, device):
        if "NPU" in device:
            with open(os.path.join(model, f"{model_name}.blob"), "rb") as f:
                return self.core.import_model(f.read(), device)
        return self.core.compile_model(os.path.join(model, f"{model_name}.xml"), device)
    

    def __call__(
            self,
            prompt,
            image: PIL.Image.Image = None,
            mask_image: PIL.Image.Image = None,
            negative_prompt=None,
            scheduler=None,
            strength = 0.5,
            num_inference_steps = 32,
            guidance_scale = 7.5,
            eta = 0.0,
            create_gif = False,
            model = None,
            callback = None,
            callback_userdata = None
    ):
        # extract condition
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_embeddings = self.text_encoder(text_input.input_ids)[self._text_encoder_output]
    

        # do classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
        
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
                
            tokens_uncond = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length, #truncation=True,  
                return_tensors="np"
            )
            uncond_embeddings = self.text_encoder(tokens_uncond.input_ids)[self._text_encoder_output]
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, scheduler)
        latent_timestep = timesteps[:1]
        
        print("---Before prepare mask and masked image size---", image.size)
        print("---Before prepare mask and masked MASK size---", mask_image.size)

        #preprocess image and mask
        mask, masked_image, init_image = prepare_mask_and_masked_image(
            image, mask_image, self.height, self.width, return_image=True)
        print("After prepare mask and masked image", masked_image.shape)
        print("before prepare mask latents")     
        
        mask, masked_image_latents = self.prepare_mask_latents(mask, masked_image, do_classifier_free_guidance)

        print("After prepare mask")
        # get the initial random noise unless the user supplied it
        latents = self.prepare_latents(init_image, latent_timestep, scheduler)


        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        if create_gif:
            frames = []        

        for i, t in enumerate(self.progress_bar(timesteps)):
            if callback:
               callback(i, callback_userdata)

            # expand the latents if we are doing classifier free guidance
            #latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            #latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = []
            latent_model_input = latents 
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            #print("loop latent_model_input:", latent_model_input.shape)
            #print("loop mask:", mask.shape)
            #print("loop masked_image_latents:", masked_image_latents.shape)
            latent_model_input = np.concatenate([latent_model_input, mask, masked_image_latents], axis=1)
            
            
            time_proj_dict = {"t" : np.expand_dims(np.float32(t), axis=0)}
            self.infer_request_time_proj.start_async(time_proj_dict)
            self.infer_request_time_proj.wait()
            time_proj = self.infer_request_time_proj.get_output_tensor(0).data.astype(np.float32)     

            
            # predict the noise residual
            #noise_pred = self.unet([latent_model_input, float(t), text_embeddings])[self._unet_output]
            
            input_dict_neg = {"latent_model_input":latent_model_input, "encoder_hidden_states": np.expand_dims(text_embeddings[0], axis=0), "time_proj": np.float32(time_proj)}
            self.infer_request_neg.start_async(input_dict_neg)
       
            input_dict = {"latent_model_input":latent_model_input, "encoder_hidden_states": np.expand_dims(text_embeddings[1], axis=0), "time_proj": np.float32(time_proj)}
            self.infer_request.start_async(input_dict)
            
            self.infer_request_neg.wait()
            self.infer_request.wait()
            noise_pred_neg = self.infer_request_neg.get_output_tensor(0)
            noise_pred_pos = self.infer_request.get_output_tensor(0)


            noise_pred.append(noise_pred_neg.data.astype(np.float32))
            noise_pred.append(noise_pred_pos.data.astype(np.float32)) 
            #print("noise_pred:",noise_pred)
            # perform guidance
            if do_classifier_free_guidance:
                #noise_pred_uncond = negative, noise_pred_text = positive
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_diff = noise_pred_text - noise_pred_uncond
                noise_pred = noise_pred_uncond + guidance_scale * (noise_diff)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs)["prev_sample"].numpy()
     
            if create_gif:
                frames.append(latents)
              
        if callback:
            callback(num_inference_steps, callback_userdata)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        
        image = self.vae_decoder(latents)[self._vae_d_output]
      
        image = self.postprocess_image(image)

        if create_gif:
            gif_folder=os.path.join(model,"../../../gif")
            if not os.path.exists(gif_folder):
                os.makedirs(gif_folder)
            for i in range(0,len(frames)):
                image = self.vae_decoder(frames[i])[self._vae_d_output]
                image = self.postprocess_image(image)
                output = gif_folder + "/" + str(i).zfill(3) +".png"
                cv2.imwrite(output, image)
            with open(os.path.join(gif_folder, "prompt.json"), "w") as file:
                json.dump({"prompt": prompt}, file)
            frames_image =  [Image.open(image) for image in glob.glob(f"{gif_folder}/*.png")]  
            frame_one = frames_image[0]
            gif_file=os.path.join(gif_folder,"stable_diffusion.gif")
            frame_one.save(gif_file, format="GIF", append_images=frames_image, save_all=True, duration=100, loop=0)

        return image
    
    def prepare_latents(self, input_image:PIL.Image.Image = None, latent_timestep:torch.Tensor = None, scheduler = LMSDiscreteScheduler):
        """
        Function for getting initial latents for starting generation
        
        Parameters:
            image (PIL.Image.Image, *optional*, None):
                Input image for generation, if not provided randon noise will be used as starting point
            latent_timestep (torch.Tensor, *optional*, None):
                Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
        Returns:
            latents (np.ndarray):
                Image encoded in latent space
        """
        latents_shape = (1, 4, self.height // 8, self.width // 8)
   
        noise = np.random.randn(*latents_shape).astype(np.float32)
        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        
        if input_image is None:
            #print("Image is NONE")
            
            if isinstance(scheduler, LMSDiscreteScheduler):
                
                noise = noise * scheduler.sigmas[0].numpy()
                return noise
            elif isinstance(scheduler, EulerDiscreteScheduler):
                
                noise = noise * scheduler.sigmas.max().numpy()
                return noise
            else:
                noise = noise * scheduler.init_noise_sigma
                return noise
    
        
        moments = self.vae_encoder(input_image)[self._vae_e_output]
      
        mean, logvar = np.split(moments, 2, axis=1)
  
        std = np.exp(logvar * 0.5)
        latents = (mean + std * np.random.randn(*mean.shape)) * 0.18215
       
         
        latents = scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
        return latents        

    def prepare_mask_latents(self, mask = None, masked_image = None, do_classifier_free_guidance = True):
         mask = torch.nn.functional.interpolate(mask, size=(self.height // 8, self.width // 8)).numpy()                                        
         moments = self.vae_encoder(masked_image)[self._vae_e_output] 
         mean, logvar = np.split(moments, 2, axis=1) 
         std = np.exp(logvar * 0.5)
         masked_image_latents = (mean + std * np.random.randn(*mean.shape)) * 0.18215
         mask = mask #np.concatenate([mask] * 2) if do_classifier_free_guidance else mask
         masked_image_latents = masked_image_latents #np.concatenate([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
         return mask, masked_image_latents

    def postprocess_image(self, image:np.ndarray):
        """
        Postprocessing for decoded image. Takes generated image decoded by VAE decoder, unpad it to initila image size (if required), 
        normalize and convert to [0, 255] pixels range. Optionally, convertes it from np.ndarray to PIL.Image format
        
        Parameters:
            image (np.ndarray):
                Generated image
        Returns:
            image (List of np.ndarray or PIL.Image.Image):
                Postprocessed images

                        if "src_height" in meta:
            orig_height, orig_width = meta["src_height"], meta["src_width"]
            image = [cv2.resize(img, (orig_width, orig_height))
                        for img in image]
    
        return image
        """
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = (image[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
                        
        return image


    def get_timesteps(self, num_inference_steps:int, strength:float, scheduler):
        """
        Helper function for getting scheduler timesteps for generation
        In case of image-to-image generation, it updates number of steps according to strength
        
        Parameters:
           num_inference_steps (int):
              number of inference steps for generation
           strength (float):
               value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. 
               Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input.
        """
        # get the original timestep using init_timestep
   
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start 
