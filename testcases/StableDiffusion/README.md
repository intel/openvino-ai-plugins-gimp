# Running Stable Diffusion without Installing GIMP
 
For these instructions, we assume that you have cloned this repo inside `C:\Users\Public\GIMP` directory and that the virtual environment for GIMP is in `C:\Users\Public\GIMP\gimpenv3`. 
## Install and Model Setup

The following commands can be run in a new command window.
1. Navigate to C:\users\Public\GIMP
  ```
  cd  C:\Users\Public\GIMP
  ```
  
2. Run install.bat from C:\Users\Public\GIMP directory.  

```
openvino-ai-plugins-gimp\install.bat --install_models
```
3. As install.bat completes, you will be asked to download the models with a few prompts. There are now two options: 

    Download the models for the first time, follow A. 

    Recompile already downloaded models, follow B. 
    
    A.  Downloading for the first time <br/>
    Download "1", Stable Diffusion 1.5 Square as an instance.  <br/>

    ![](../../Docs/figs/standalone1.png) 

    B. Already installed but needs recompiling.  <br/>
    If you have already downloaded the models, it will show as Installed. If you need to recompile just choose it again <br/>

    ![](../../Docs/figs/standalone2.png) 

4. Activate the environment. 
  ```
  gimpenv3\Scripts\activate 
  ```
## Running the Stable Diffusion Commandline

These commands should be run in the same window where you have activated the environment. <br>
Note: There are a couple warnings printed which can be safely ignored. 

### Basic Options
Run the python stable diffusion engine with the -h option to see the possible options.
```
python openvino-ai-plugins-gimp\testcases\StableDiffusion\stable_diffusion_engine_tc.py -h
``` 
You should see the following output:
  ![](../../Docs/figs/standalone3.png) 

### Example 1: Defaults 
Default options for INT8 on NPU
```
python openvino-ai-plugins-gimp\testcases\StableDiffusion\stable_diffusion_engine_tc.py -m sd_1.5_square_int8 -pm "best power efficiency"
```
  ![](../../Docs/figs/standalone4.png) 

In the output above, you should see the iterations per second. This is an instantaenous measurement, and not an average. It gives you a general idea of the speed of Unet, but the more important number is the Image generation time (red arrow at the bottom).
 
### Example 2: Overriding the devlces
Command line options allow you to over ride the devices in config.json without having to modify the config. For example, moving the Text Encoder to GPU in the power efficiency mode:
```
python openvino-ai-plugins-gimp\testcases\StableDiffusion\stable_diffusion_engine_tc.py -m sd_1.5_square_int8 -pm "best power efficiency" -td GPU
```
![](../../Docs/figs/standalone5.png) 

### Example 3: Generating multiple images.
In general, the first inference is slower. The recommendation is to generate multiple images in a row. The script will output the average image generation time.
```
python openvino-ai-plugins-gimp\testcases\StableDiffusion\stable_diffusion_engine_tc.py -m sd_1.5_square_int8 -pm "best power efficiency" -td GPU -n 5
```
![](../../Docs/figs/standalone6.png) 

### Example 4: Saving images to check accuracy
Use the -si option to save images in the directory that the script is run from. 
```
python openvino-ai-plugins-gimp\testcases\StableDiffusion\stable_diffusion_engine_tc.py -m sd_1.5_square_int8 -pm "best power efficiency" -td GPU -n 5 -si
```
![](../../Docs/figs/standalone7.png) 


