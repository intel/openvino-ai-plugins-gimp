# Image generation with Stable Diffusion 3.0

Stable Diffusion V3 is next generation of latent diffusion image Stable Diffusion models family that outperforms state-of-the-art text-to-image generation systems in typography and prompt adherence, based on human preference evaluations. 

In comparison with previous versions, it based on Multimodal Diffusion Transformer (MMDiT) text-to-image model that features greatly improved performance in image quality, typography, complex prompt understanding, and resource-efficiency.

More details about model can be found in [model card](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [research paper](https://stability.ai/news/stable-diffusion-3-research-paper) and [Stability.AI blog post](https://stability.ai/news/stable-diffusion-3-medium).

## Enabling SD3 in GIMP
### Converting the Models
The supported version of SD3 in GIMP is the flash_sd3 LoRA. In order to use with GIMP, you first need to download and convert the model into OpenVINO format. This can be done by following the instructions in the [OpenVINO Stable Diffusion v3 Notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/stable-diffusion-v3).
>**Note**: Ensure you use the instructions for flash_sd3. 

### Model Installation
After completing the steps in the notebook, you will have the models located in `openvino_notebooks\notebooks\stable-diffusion-v3\stable-diffusion-3`. Copy these models to `openvino-ai-plugins-gimp\weights\stable-diffusion-ov\stable-diffusion-3.0` 

For example, using robocopy to copy the files on the command line:
```
robocopy openvino_notebooks\notebooks\stable-diffusion-v3\stable-diffusion-3\ %userprofile%\openvino-ai-plugins-gimp\weights\stable-diffusion-ov\stable-diffusion-3.0\. /mir
```
After copying, create a file called  `install_info.json` inside the stable-diffusion-3.0 directory. This will ensure that the GIMP plugin will recognize this as a valid model. It should contain the following text:
```
{
    "hf_repo_id": "none",
    "hf_commit_id": "none"
}
```

Verify that the copied files and directory structure looks as follows:

![image](https://github.com/user-attachments/assets/039073d1-e593-4365-92c9-3555ea023670)
### Running with GIMP
After completing model installation steps, SD3 will now be available in the Stable Diffusion UI. Note that SD3 can generate valid images in as few as 4 iterations. Also, the guidance scale needed is normally much lower. See the screenshot below:
![image](https://github.com/user-attachments/assets/6daf3201-b873-4198-a752-19594f352c50)




