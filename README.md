

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
2. Clone, run install script and copy the stable-diffusion-1.5 models for NPU: <br>

   - clone this repo: <br>
   ```git clone https://github.com/intel/openvino-ai-plugins-gimp.git``` <br>
   
    - run install script -> this will create the virtual environment "gimpenv3", install all required packages and will also walk you through models setup. <br>
   ```openvino-ai-plugins-gimp\install.bat``` <br>
   
   - Download & extract custom NPU compatible SD-1.5 models from - [stable-diffusion-1.5-internal-blobs-NEW](https://intel.sharepoint.com/:u:/r/sites/NPUIPProductPlanningExecution-CustomerSolutionsEngineering/Shared%20Documents/Customer%20Solutions%20Engineering/Demo_Material/Vision/2023/Demo%20Installers/GIMP-Optimized-min16gb_RAM/stable-diffusion-1.5-internal-blobs-NEW.zip?csf=1&web=1&e=DEclbr) and put the folder stable-diffusion-1.5-internal-blobs-NEW in the following location: <br>
    ```C:\Users<user_name>\openvino-ai-plugins-gimp\weights\stable-diffusion-ov``` <br>
    
   
3. Start the GIMP application, and add the gimpenv3 path that was printed when running the above step to the list of plugin folders [Edit-> Preferences-> Folders-> Plugins]. <br>
   Example:  ```Plug-ins in GIMP :  <path\to>\gimpenv3\lib\site-packages\gimpopenvino\plugins``` Add this path to [Edit-> Preferences-> Folders-> Plugins] in GIMP <br>
4. Restart GIMP, and you should see 'OpenVINO-AI-Plugins' show up in 'Layer' menu <br>


### OpenVINO™ Image Generator Plugin with Stable Diffusion - This GIF doesn't represent the current GUI
1. Create or choose a layer  <br>
2. Select Stable Diffusion from the drop down list in layers -> OpenVINO-AI-Plugins <br>
3. Choose the desired model and device from the drop down list.<br>
4. For running on GPU & NPU - Select stable-diffusion-1.5-internal-blobs-NEW from the model drop down list and choose Text_encoder Device - CPU, Unet Device - GPU/NPU, UnetNeg Device - NPU/GPU, VAE Device - GPU <br>
6. Click on "Load Models" to compile & load the model on the selected device. Wait for it to complete. Please note that you need to perform this step only if you change the model or device or both. For any subsequent runs just click "Run Inference" <br>
7. Enter prompt and other parameters (select EulerDiscrete for scheduler) <br>
8. Click on “Run Inference”. Wait for the total inference steps to get completed. <br>
9. If create gif option is selected, please note that performance will reduce. The generated gif is located in below path. You can play it in GIMP by going to Filters -> Animations -> Playback <br>
```C:\Users\<user_name>\openvino-ai-plugins-gimp\gif\stable_diffusion.gif``` <br>

![](gifs/stable-diffusion.png)
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

