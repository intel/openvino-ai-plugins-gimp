

# OpenVINO™ Plugins for GIMP

This branch is under development. <br>Dedicated for GIMP 3, Python 3 and OpenVino.<br> :star: :star: :star: :star: are welcome.<br>

## Current list of plugins:
[1] Super-Resolution
[2] Style-Transfer
[3] Inpainting
[4] Semantic-Segmentation
[5] Stable-Diffusion

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
import gimpov
```

# Use with GIMP
![gimp-screenshot](gimp-screenshot.PNG)

## Installation Steps
[1] Install [GIMP](https://www.gimp.org/downloads/devel/) 2.99.6  (Only windows and linux) <br>
[2] Clone this repository: git clone https://github.com/intel-sandbox/GIMP-ML-OV/tree/openvino-gimp <br>
[3] Change branch : <br>
```git checkout --track origin/GIMP3-ML``` <br>
[3] On linux, run for GPU/CPU: <br>
```bash GIMP-ML/install.bat```<br>
On windows, run for CPU: <br>
```GIMP-ML\install.bat```<br>
On windows, run for GPU: <br>
```GIMP-ML\install.bat gpu```<br>
[4] Follow steps that are printed in terminal or cmd. <br>


# Acknowledgements
* Plugin architecture inspired from GIMP-ML https://github.com/kritiksoman/GIMP-ML/tree/GIMP3-ML
* Stable Diffusion Engine https://github.com/bes-dev/stable_diffusion.openvino



# License
#TODO


# Disclaimer
The authors are not responsible for the content generated using this project. Please, don't use this project to produce illegal, harmful, offensive etc. content.
