

# OpenVINO™ Plugins for GIMP

This branch is under development. <br>Dedicated for GIMP 3, Python 3 and OpenVino.<br> :star: :star: :star: :star: are welcome.<br>

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


# Use as a Python Package
```Python
import cv2
import gimpopenvino
```

# Use with GIMP
![gimp-screenshot](gimp-screenshot.PNG)

## Installation Steps
[1] Install [GIMP](https://www.gimp.org/downloads/devel/) 2.99.6  (Only windows and linux) <br>
[2] Clone this repository: git clone https://github.com/intel-sandbox/GIMP-ML-OV.git <br>
[3] Rename the repository : <br>
```mv GIMP-ML-OV GIMP-OpenVino``` <br>
[3] On linux, run for GPU/CPU: <br>
```bash GIMP-OpenVINO/install.bat```<br>
On windows, run for CPU: <br>
```GIMP-OpenVINO\install.bat```<br>
[4] Follow steps that are printed in terminal or cmd. <br>
[5] Copy the weights folder to ```C:\Users\<user_name>\GIMP-OpenVINO\weights``` <br>
[6] Download Stable-Diffusion models from https://huggingface.co/bes-dev/stable-diffusion-v1-4-openvino/tree/main and place it in ```C:\Users\<user_name>\GIMP-OpenVINO\weights\stable-diffusion-ov```


# Acknowledgements
* Plugin architecture inspired from GIMP-ML - https://github.com/kritiksoman/GIMP-ML/tree/GIMP3-ML
* Stable Diffusion Engine - https://github.com/bes-dev/stable_diffusion.openvino



# License
Apache 2.0


# Disclaimer
Stable Diffusion’s data model is governed by the Creative ML Open Rail M license, which is not an open source license.
https://github.com/CompVis/stable-diffusion. Users are responsible for their own assessment whether their proposed use of the project code and model would be governed by and permissible under this license.

