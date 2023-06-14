

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
2. Clone, run install script: <br>

   - clone this repo: <br>
   ```git clone https://github.com/intel/openvino-ai-plugins-gimp.git``` <br>
   
    - run install script - this will create the virtual environment "gimpenv3", install all required packages and will also walk you through models setup. <br>
   ```openvino-ai-plugins-gimp\install.bat``` <br>
   
3. Start the GIMP application, and add the gimpenv3 path that was printed when running the above step to the list of plugin folders [Edit-> Preferences-> Folders-> Plugins]. <br>
   Example:  ```Plug-ins in GIMP :  <path\to>\gimpenv3\lib\site-packages\gimpopenvino\plugins``` Add this path to [Edit-> Preferences-> Folders-> Plugins] in GIMP <br>
4. Restart GIMP, and you should see 'OpenVINO-AI-Plugins' show up in 'Layer' menu <br>

### Linux (Tested on Ubuntu 22.04)
1. Install flatpak distribution of GIMP development package -- instructions here: [GIMP Development Downloads](https://www.gimp.org/downloads/devel/) <br>
2. Clone, run install script, copy weights: <br>

   ```
   # clone this repo:
   git clone https://github.com/intel/openvino-ai-plugins-gimp.git 

   # run install script - this will create the virtual environment "gimpenv3", install all required packages and will also walk you through models setup 
   openvino-ai-plugins-gimp/install.sh
   ```
3. Start the GIMP application (```flatpak run org.gimp.GIMP```), and add the gimpenv3 path that was printed when running the above step to the list of plugin folders  [Edit-> Preferences-> Folders-> Plugins]. <br>
4. Restart GIMP, and you should see 'OpenVINO-AI-Plugins' show up in 'Layer' menu <br>

### OpenVINO™ Image Generator Plugin with Stable Diffusion - This GIF doesn't represent the current GUI
1. Create or choose a layer  <br>
2. Select Stable Diffusion from the drop down list in layers -> OpenVINO-AI-Plugins <br>
3. Choose the desired model and device from the drop down list.<br>
4. Click on "Load Models" to compile & load the model on the selected device. Wait for it to complete. Please note that you need to perform this step only if you change the model or device or both. For any subsequent runs just click "Run Inference" <br>
3. Enter prompt and other parameters <br>
4. Click on “Run Inference”. Wait for the total inference steps to get completed. <br>
6. If create gif option is selected, please note that performance will reduce. The generated gif is located in below path. You can play it in GIMP by going to Filters -> Animations -> Playback <br>
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

