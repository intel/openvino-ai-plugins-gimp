#steps:
#1: python -m venv model_conv
#2: model_conv\Scripts\activate
#3: python -m pip install --upgrade pip wheel setuptools
#4: pip install -r model-requirements.txt
#5: python sd_model_conversion.py 2022.3.0




from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import gc
from pathlib import Path
import torch
from torch.onnx import _export as torch_onnx_export
import numpy as np
import sys
import os
import shutil
from openvino.tools import mo
from openvino.runtime import serialize
import platform
import subprocess
from controlnet_aux import OpenposeDetector
from diffusers import UniPCMultistepScheduler
from huggingface_hub import hf_hub_download



#ov_version = sys.argv[1]

#print("ov_version",ov_version)
install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp")
SD_path = os.path.join(install_location, "weights", "stable-diffusion-ov")

if platform.system() == "Linux":
	sd_mo_path=os.path.join(".", "model_conv/bin/mo")
else:
	sd_mo_path=r'model_conv\Scripts\mo.exe'

choice = sys.argv[1]

#if ov_version == "2022.3.0":




wt = 512
ht = 512
channel = 4
  
if choice == "7":
    print("============SD-1.5 Controlnet-Openpose Model setup============----")
    wt = 512
    ht = 512
    weight_path = os.path.join(SD_path, "controlnet-openpose")
    
       
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet)
    scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.save_config(os.path.join(SD_path,"UniPCMultistepScheduler_config"))


  
    lllyasviel_ControlNet_path = os.path.join(weight_path, "lllyasviel_ControlNet")

    if os.path.isdir(lllyasviel_ControlNet_path):
     shutil.rmtree(lllyasviel_ControlNet_path)

    repo_id="lllyasviel/Annotators"
 
    file = hf_hub_download(repo_id=repo_id, filename="body_pose_model.pth")
    hf_hub_download(repo_id=repo_id, filename="hand_pose_model.pth")
    hf_hub_download(repo_id=repo_id, filename="facenet.pth")
    download_folder = os.path.join(file, "..")
    print("download_folder", download_folder)
    
    shutil.copytree(download_folder, lllyasviel_ControlNet_path)
    
    
    pose_estimator = OpenposeDetector.from_pretrained(lllyasviel_ControlNet_path) 
    
    
    OPENPOSE_ONNX_PATH = Path(weight_path) / 'openpose.onnx'
    OPENPOSE_OV_PATH = Path(weight_path) / 'openpose.xml'  

    if not OPENPOSE_OV_PATH.exists():
        if not OPENPOSE_ONNX_PATH.exists():
            torch.onnx.export(pose_estimator.body_estimation.model, torch.zeros([1, 3, 184, 136]), OPENPOSE_ONNX_PATH)
        try:
            print("---In TRY----")
            openpose_model = mo.convert_model(OPENPOSE_ONNX_PATH, compress_to_fp16=True)
            serialize(openpose_model, xml_path=os.path.join(weight_path, 'openpose.xml'))
          
        except:
            print("---In except----")
            subprocess.call([sd_mo_path, '--input_model', OPENPOSE_ONNX_PATH, '--data_type=FP16', '--output_dir', weight_path])
            
        print('OpenPose successfully converted to IR')
    else:
        print(f"OpenPose will be loaded from {OPENPOSE_OV_PATH}") 

    if not os.path.isdir(weight_path):
            os.makedirs(weight_path)

    print("weight path is :", weight_path)


    CONTROLNET_ONNX_PATH = Path(weight_path) / 'controlnet-pose.onnx'
    CONTROLNET_OV_PATH = Path(weight_path) / 'controlnet-pose.xml'        

  
if choice == "8":
    print("============SD-1.5 Controlnet-Canny Model setup============----")
    wt = 512
    ht = 512
    weight_path = os.path.join(SD_path, "controlnet-canny")
    
       
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet
    )
    
    
    if not os.path.isdir(os.path.join(SD_path,"UniPCMultistepScheduler_config")):
        print("Saving scheduler config") # TODO: change
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet)
        scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        scheduler.save_config(os.path.join(SD_path,"UniPCMultistepScheduler_config"))               
    
    if not os.path.isdir(weight_path):
            os.makedirs(weight_path)

    print("weight path is :", weight_path)


    CONTROLNET_ONNX_PATH = Path(weight_path) / 'controlnet-canny.onnx'
    CONTROLNET_OV_PATH = Path(weight_path) / 'controlnet-canny.xml'     

else:
    print("============Select option 7============----")
    
  

    




UNET_ONNX_PATH = Path(weight_path) / 'unet_controlnet' / 'unet_controlnet.onnx'
UNET_OV_PATH = Path(weight_path) / 'unet_controlnet.xml'

print("UNET PATH",UNET_OV_PATH)
print("UNET_ONNX_PATH", UNET_ONNX_PATH)


inputs = {
    "sample": torch.randn((2, 4, 64, 64)),
    "timestep": torch.tensor(1),
    "encoder_hidden_states": torch.randn((2,77,768)),
    "controlnet_cond": torch.randn((2,3,512,512))
}
controlnet.eval()
with torch.no_grad():
    down_block_res_samples, mid_block_res_sample = controlnet(**inputs, return_dict=False)
controlnet_output_names = [f"down_block_res_sample_{i}" for i in range(len(down_block_res_samples))]
controlnet_output_names.append("mid_block_res_sample")
if not CONTROLNET_OV_PATH.exists():
    if not CONTROLNET_ONNX_PATH.exists() :

        with torch.no_grad():
            torch_onnx_export(controlnet, inputs, CONTROLNET_ONNX_PATH, input_names=list(inputs), output_names=controlnet_output_names, onnx_shape_inference=False)
            
    try:
        controlnet_model = mo.convert_model(CONTROLNET_ONNX_PATH, compress_to_fp16=True)
        serialize(controlnet_model, xml_path=os.path.join(weight_path, 'controlnet-pose.xml'))
    except:
        subprocess.call([sd_mo_path, '--input_model', CONTROLNET_ONNX_PATH, '--data_type=FP16', '--output_dir', weight_path])
    

    print('ControlNet successfully converted to IR')
else:
    print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")
    
if not UNET_OV_PATH.exists():
    if not UNET_ONNX_PATH.exists():
        UNET_ONNX_PATH.parent.mkdir(exist_ok=True)
        inputs.pop("controlnet_cond", None)
        inputs["down_block_additional_residuals"] = down_block_res_samples
        inputs["mid_block_additional_residual"] = mid_block_res_sample

        unet = pipe.unet
        unet.eval()

        input_names = ["sample", "timestep", "encoder_hidden_states", *controlnet_output_names]

        with torch.no_grad():
            torch_onnx_export(unet, inputs, str(UNET_ONNX_PATH), input_names=input_names, output_names=["sample_out"], onnx_shape_inference=False)
        del unet
    del pipe.unet
    gc.collect()
    
    try:
        unet_model = mo.convert_model(UNET_ONNX_PATH, compress_to_fp16=True)
        serialize(unet_model, xml_path=os.path.join(weight_path, 'unet_controlnet.xml'))
    except:
        subprocess.call([sd_mo_path, '--input_model', UNET_ONNX_PATH, '--data_type=FP16', '--output_dir', weight_path]) 
        

    print('Unet successfully converted to IR')
else:
    del pipe.unet
    print(f"Unet will be loaded from {UNET_OV_PATH}")
gc.collect()
    
    
    
TEXT_ENCODER_ONNX_PATH = Path(weight_path) / 'text_encoder.onnx'
TEXT_ENCODER_OV_PATH = Path(weight_path) / 'text_encoder.xml'
print("TEXT_ENCODER_OV_PATH:",TEXT_ENCODER_OV_PATH)  


def convert_encoder_onnx(text_encoder: torch.nn.Module, onnx_path:Path):
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
            torch_onnx_export(
                text_encoder,  # model instance
                input_ids,  # inputs for model tracing
                onnx_path,  # output file for saving result
                input_names=['tokens'],  # model input name for onnx representation
                output_names=['last_hidden_state', 'pooler_out'],  # model output names for onnx representation
                opset_version=14,  # onnx opset version for export
                onnx_shape_inference=False
            )
        print('Text Encoder successfully converted to ONNX')
    

if not TEXT_ENCODER_OV_PATH.exists(): #--compress_to_fp16 --data_type=FP16
    
    convert_encoder_onnx(pipe.text_encoder, TEXT_ENCODER_ONNX_PATH)
    try:
        encoder_model = mo.convert_model(TEXT_ENCODER_ONNX_PATH, compress_to_fp16=True)
        serialize(encoder_model, xml_path=os.path.join(weight_path, 'text_encoder.xml'))
  
    except:
        subprocess.call([sd_mo_path, '--input_model', TEXT_ENCODER_ONNX_PATH, '--data_type=FP16', '--output_dir', weight_path])
          
    print('Text Encoder successfully converted to IR')
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")

del pipe.text_encoder
gc.collect()

#if ov_version == "2022.2.0":


VAE_DECODER_ONNX_PATH = Path(weight_path) /'vae_decoder.onnx'
VAE_DECODER_OV_PATH = Path(weight_path) / 'vae_decoder.xml'


def convert_vae_decoder_onnx(vae: torch.nn.Module, onnx_path: Path):
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
            
            return self.vae.decode(latents)

    if not onnx_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, ht//8, wt//8))

        vae_decoder.eval()
        with torch.no_grad():
            torch.onnx.export(vae_decoder, latents, onnx_path, input_names=[
                              'latents'], output_names=['sample'])
        print('VAE decoder successfully converted to ONNX')


if not VAE_DECODER_OV_PATH.exists():
    convert_vae_decoder_onnx(pipe.vae, VAE_DECODER_ONNX_PATH)
 
    try:
        vae_decoder_model = mo.convert_model(VAE_DECODER_ONNX_PATH, compress_to_fp16=True)
        serialize(vae_decoder_model, xml_path=os.path.join(weight_path, 'vae_decoder.xml'))
    except:
        subprocess.call([sd_mo_path, '--input_model', VAE_DECODER_ONNX_PATH, '--data_type=FP16', '--output_dir', weight_path])
       
    print('VAE decoder successfully converted to IR')
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")

del pipe.vae


#cleanup
if TEXT_ENCODER_ONNX_PATH.exists():
    os.remove(TEXT_ENCODER_ONNX_PATH)
    
if UNET_ONNX_PATH.exists():
    shutil.rmtree(Path(weight_path) / 'unet_controlnet')
    
if VAE_DECODER_ONNX_PATH.exists():
    os.remove(VAE_DECODER_ONNX_PATH)

    
sys.exit()
