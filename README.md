

# OpenVINO™ AI Plugins for GIMP

This branch is currently under development. <br>Dedicated for GIMP 3, Python 3 and OpenVino.<br> :star: :star: :star: :star: are welcome.<br>

## Current list of plugins:
[1] Super-Resolution <br>
[2] Style-Transfer <br>
[3] Inpainting <br>
[4] Semantic-Segmentation <br>
[5] Stable-Diffusion <br>

# Objectives
[1] Provides a set of OpenVino based plugins that add AI features to GIMP. <br>
[2] Serve as a refrence code for how to make use of OpenVino in GIMP application for inferencing on Intel's' CPU & GPU  <br>
[3] Add AI to routine image editing workflows. <br>

# Contribution 
Welcome people interested in contribution !! 
Please raise a PR for any new features, modifactions or bug fixes. 

# Use with GIMP
![gimp-screenshot](gimp-screenshot.PNG)

## Installation Steps

### Windows
1. Install [GIMP 2.99.10 (revision 2)](https://download.gimp.org/gimp/v2.99/windows/gimp-2.99.10-setup-2.exe) or Install [GIMP 2.99.14](https://download.gimp.org/gimp/v2.99/windows/gimp-2.99.14-setup.exe) <br>
2. Clone, run install script, copy weights:
   ```
   :: clone this repo:
   git clone https://github.com/intel/openvino-ai-plugins-gimp.git
   
   :: run install script - this will create the virtual environment "gimpenv3" and install all required packages.
   openvino-ai-plugins-gimp\install.bat
   
   :: Copy the openvino models to user weights folder
   Xcopy /E /I .\openvino-ai-plugins-gimp\weights %userprofile%\openvino-ai-plugins-gimp\weights\
   ```
4. Download Stable-Diffusion-1.4 models from https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/tree/main and place it in <br> 
```%userprofile%\openvino-ai-plugins-gimp\weights\stable-diffusion-ov\stable-diffusion-1.4``` <br>
5. Start the GIMP application, and add the gimpenv3 path that was printed at the end of the install script to the list of plugin folders [Edit-> Preferences-> Folders-> Plugins]. <br>
6. Restart GIMP, and you should see 'OpenVINO-AI-Plugins' show up in 'Layer' menu <br>

### Generate Stable-Diffusion-1.5 openvino model -- NEW !! (Still in active development)
1. python -m venv model_conv <br>
2. model_conv\Scripts\activate <br>
3. python -m pip install --upgrade pip wheel setuptools <br>
4. cd openvino-ai-plugins-gimp
5. pip install -r model-requirements.txt  --> (Modify the openvino version if needed) <br>
6. python sd_model_conversion.py <br>

### Running Stable Diffusion model Server -- NEW !! (Still in active development)
1. Open a new command window  <br>
2. Run Stable Diffusion model Server : This is done to reduce the start-up latency in loading the models onto the devices & importing python packages upfront. <br>
   ```<path\to\your>\gimpenv3\Scripts\python.exe <path\to\your\openvino-ai-plugins-gimp\gimpopenvino\tools\stable-diffusion-ov-server.py``` <br>
   You will see the models being loaded and compiled, once its done "Waiting first" will be printed <br>
   - In order to change the devices or switch between models, please modify L167/L168 in openvino-ai-plugins-gimp\gimpopenvino\tools\stable-diffusion-ov-server.py and restart the server <br>

### OpenVINO™ Image Generator Plugin with Stable Diffusion - This GIF doesnt represent the current GUI
1. Create or choose a layer  <br>
2. Select Stable Diffusion from the drop down list in layers -> OpenVINO-AI-Plugins <br>
3. Enter a prompt and other parameters <br>
4. Click on “Run Inference”. Wait for the total inference steps to get completed. (Can be viewed in Stable Diffusio Server output window) <br>
6. If create gif option is selected, please note that performance will reduce. The generated gif is located in below path. You can play it in GIMP by going to Filters -> Animations -> Playback <br>
```C:\Users\<user_name>\openvino-ai-plugins-gimp\gif\stable_diffusion.gif``` <br>

![](gifs/stable-diffusion.webp)


### OpenVINO™ Semantic Segmentation Plugin
![](gifs/semantic-segmentation.webp)

### OpenVINO™ Super Resolution Plugin 
![](gifs/super-res.webp)

### OpenVINO™ Style Transfer Plugin
![](gifs/style-transfer.webp)

### OpenVINO™ Inpainting Plugin 
1. Open an image in GIMP. <br>
2. Make sure there is alpha channel added to the image by right clicking on the image from layer section and selecting “Add alpha channel” <br>
3. Add a new transparent layer of the same size as original image. <br>
4. Select paint brush with white foreground color and black background color. Choose the thickness of the brush <br>
10. Now paint the object that you want to remove from the image. <br>
11. Select the new layer and image at the same. You should see “two items selected in layer section” <br>


![](gifs/inpainting.webp)





# Acknowledgements
* Plugin architecture inspired from GIMP-ML - https://github.com/kritiksoman/GIMP-ML/tree/GIMP3-ML
* Stable Diffusion Engine - https://github.com/bes-dev/stable_diffusion.openvino



# License
Apache 2.0


# Disclaimer
Stable Diffusion’s data model is governed by the Creative ML Open Rail M license, which is not an open source license.
https://github.com/CompVis/stable-diffusion. Users are responsible for their own assessment whether their proposed use of the project code and model would be governed by and permissible under this license.

