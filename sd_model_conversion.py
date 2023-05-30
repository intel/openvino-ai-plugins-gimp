#steps:
#1: python -m venv model_conv
#2: model_conv\Scripts\activate
#3: python -m pip install --upgrade pip wheel setuptools
#4: pip install -r model-requirements.txt
#5: python sd_model_conversion.py 2022.3.0




from diffusers import StableDiffusionPipeline
import gc
from pathlib import Path
import torch
import numpy as np
import sys
import os

#ov_version = sys.argv[1]

#print("ov_version",ov_version)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")

text_encoder = pipe.text_encoder
text_encoder.eval()
unet = pipe.unet
unet.eval()
vae = pipe.vae
vae.eval()

del pipe

install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp")
weight_path = os.path.join(install_location, "weights\stable-diffusion-ov\stable-diffusion-1.5")

if not os.path.isdir(weight_path):
        os.mkdir(weight_path)

print("weight path is :", weight_path)

#if ov_version == "2022.3.0":

TEXT_ENCODER_ONNX_PATH = Path(weight_path) / 'text_encoder.onnx'
TEXT_ENCODER_OV_PATH = TEXT_ENCODER_ONNX_PATH.with_suffix('.xml')
print("TEXT_ENCODER_ONNX_PATH:",TEXT_ENCODER_OV_PATH)


def convert_encoder_onnx(xtext_encoder: StableDiffusionPipeline, onnx_path:Path):
    """
    Convert Text Encoder model to ONNX. 
    Function accepts pipeline, prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        pipe (StableDiffusionPipeline): Stable Diffusion pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    if not onnx_path.exists():
        input_ids = torch.ones((1, 77), dtype=torch.long)
        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            # infer model, just to make sure that it works
            text_encoder(input_ids)
            # export model to ONNX format
            torch.onnx.export(
                text_encoder,  # model instance
                input_ids,  # inputs for model tracing
                onnx_path,  # output file for saving result
                input_names=['tokens'],  # model input name for onnx representation
                output_names=['last_hidden_state', 'pooler_out'],  # model output names for onnx representation
                opset_version=14  # onnx opset version for export
            )
        print('Text Encoder successfully converted to ONNX')
    

if not TEXT_ENCODER_OV_PATH.exists(): #--compress_to_fp16 --data_type=FP16
    
    convert_encoder_onnx(text_encoder, TEXT_ENCODER_ONNX_PATH)
    os.system('mo --input_model %s --compress_to_fp16 --output_dir %s' % (TEXT_ENCODER_ONNX_PATH,weight_path))
    print('Text Encoder successfully converted to IR')
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")

del text_encoder
gc.collect()

#if ov_version == "2022.2.0":

UNET_ONNX_PATH = Path(weight_path) / 'unet/unet.onnx'
UNET_OV_PATH = UNET_ONNX_PATH.parents[1] / 'unet.xml'

print("UNET PATH",UNET_OV_PATH)


def convert_unet_onnx(unet:StableDiffusionPipeline, onnx_path:Path):
    """
    Convert Unet model to ONNX, then IR format. 
    Function accepts pipeline, prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        pipe (StableDiffusionPipeline): Stable Diffusion pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    if not onnx_path.exists():
        # prepare inputs
        encoder_hidden_state = torch.ones((2, 77, 768))
        latents_shape = (2, 4, 512 // 8, 512 // 8)
        latents = torch.randn(latents_shape)
        t = torch.from_numpy(np.array(1, dtype=float))

        # model size > 2Gb, it will be represented as onnx with external data files, you will store it in separated directory for avoid a lot of files in current directory
        onnx_path.parent.mkdir(exist_ok=True, parents=True)
        unet.eval()

        with torch.no_grad():
            torch.onnx.export(
                unet, 
                (latents, t, encoder_hidden_state), str(onnx_path),
                input_names=['latent_model_input', 't', 'encoder_hidden_states'],
                output_names=['out_sample']
            )
        print('Unet successfully converted to ONNX')


if not UNET_OV_PATH.exists():
    convert_unet_onnx(unet, UNET_ONNX_PATH)
    del unet
    gc.collect()
   
    os.system('mo --input_model %s --data_type=FP16 --output_dir %s' % (UNET_ONNX_PATH,weight_path))
    print('Unet successfully converted to IR')
else:
    del unet
    print(f"Unet will be loaded from {UNET_OV_PATH}")
gc.collect()

VAE_ENCODER_ONNX_PATH = Path(weight_path) / 'vae_encoder.onnx'
VAE_ENCODER_OV_PATH = VAE_ENCODER_ONNX_PATH.with_suffix('.xml')


def convert_vae_encoder_onnx(vae: StableDiffusionPipeline, onnx_path: Path):
    """
    Convert VAE model to ONNX, then IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        pipe (StableDiffusionInstructPix2PixPipeline): InstrcutPix2Pix pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, image):
            h = self.vae.encoder(image)
            moments = self.vae.quant_conv(h)
            return moments

    if not onnx_path.exists():
        vae_encoder = VAEEncoderWrapper(vae)
        vae_encoder.eval()
        image = torch.zeros((1, 3, 512, 512))
        with torch.no_grad():
            torch.onnx.export(vae_encoder, image, onnx_path, input_names=[
                              'init_image'], output_names=['image_latent'])
        print('VAE encoder successfully converted to ONNX')


if not VAE_ENCODER_OV_PATH.exists():
    convert_vae_encoder_onnx(vae, VAE_ENCODER_ONNX_PATH)
    os.system('mo --input_model %s --compress_to_fp16 --output_dir %s' % (VAE_ENCODER_ONNX_PATH,weight_path))
    print('VAE encoder successfully converted to IR')
else:
    print(f"VAE encoder will be loaded from {VAE_ENCODER_OV_PATH}")

VAE_DECODER_ONNX_PATH = Path(weight_path) /'vae_decoder.onnx'
VAE_DECODER_OV_PATH = VAE_DECODER_ONNX_PATH.with_suffix('.xml')


def convert_vae_decoder_onnx(vae: StableDiffusionPipeline, onnx_path: Path):
    """
    Convert VAE model to ONNX, then IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        pipe (StableDiffusionInstructPix2PixPipeline): InstrcutPix2Pix pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            latents = 1 / 0.18215 * latents 
            return self.vae.decode(latents)

    if not onnx_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, 64, 64))

        vae_decoder.eval()
        with torch.no_grad():
            torch.onnx.export(vae_decoder, latents, onnx_path, input_names=[
                              'latents'], output_names=['sample'])
        print('VAE decoder successfully converted to ONNX')


if not VAE_DECODER_OV_PATH.exists():
    convert_vae_decoder_onnx(vae, VAE_DECODER_ONNX_PATH)
    os.system('mo --input_model %s --compress_to_fp16 --output_dir %s' % (VAE_DECODER_ONNX_PATH,weight_path))
    print('VAE decoder successfully converted to IR')
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")

del vae