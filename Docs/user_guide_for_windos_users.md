

# User guide for Windows Users

## Install GIMP AI plugins with OpenVINO™ Backend on GIMP

### Pre-requisite for execute on Intel NPU

>__Notes:__ To get NPU support, please following below configurations. If you are not seeking NPU version, you also can run this pulgin on any Intel CPU and GPU which OpenVINO™ is supported.

- Hardware:
  - Intel Core Ultra platform
  - 16GB system memory as minimum requirement
  - internet connection is required for installation
- Driver:
  - Intel NPU driver: 31.0.100.1688 or later versions
- Software and Package:
  - git
  - python 3.9 or 3.10
    - Note: This document will use python 3.9.13 as an example.
  - VC runtime
  - [GIMP 2.99.14](https://download.gimp.org/gimp/v2.99/windows/gimp-2.99.14-setup.exe)
  - [OpenVINO™ 2023.2](https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/windows/w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64.zip) or later version
  - [GIMP AI plugins with OpenVINO™ Backend](https://github.com/intel/openvino-ai-plugins-gimp) from Github.


### Install OpenVINO™

>__Notes:__ Use OpenVINO™ `2023.2` as an example.

Please following below steps to download and install the OpenVINO™ package.

- Check download [page](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?VERSION=v_2023_2_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE) of OpenVINO™ website, then click "Download Archives"

    ![](figs/OpenVINO_installation.png)

- Download compressed package [w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64.zip](https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/windows/w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64.zip)

    ![](figs/OpenVINO_installation_archives.png)

- Unzip `w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64.zip` to `C:\Users\Public\`

> __Notes:__ Use `C:\Users\Public\` as an example, you can unzip it to anywhere that you want.



### Install Python

>__Notes:__ Use Python `3.9.13` as an example.

Please download the prebuilt Windows x64 package from [link](https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe). After downloading, please following below steps to install the Python package.

- Check "Add Python 3.9 to PATH", then click "Install Now"

    ![](figs/python_installation_setup.png)
    ![](figs/python_installation_processing.png)


- Click "Close"

    ![](figs/python_installation_close.png)


### Install Git

>__Notes:__ Use Git `2.43.0` as an example.

Please download the prebuilt Windows x64 package from [link](https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe). After downloading, please following below steps to install the Git.

- Click "Next", then click "Install"

    ![](figs/git_installation_setup_1.png) 
    ![](figs/git_installation_setup_2.png) 
    ![](figs/git_installation_setup_3.png) 

-  Check out "View Release Notes", then click "Close"

    ![](figs/git_installation_close.png) 


### Install Microsoft Visual C++ Redistributable

Please download the latest Visual C++ Redistributable package from MSFT [site](https://aka.ms/vs/17/release/vc_redist.x64.exe). Then, install this package.

![](figs/VC_runtime_intallation.png) 
![](figs/VC_runtime_processing.png) 
![](figs/VC_runtime_close.png) 


### Install GIMP 2.99.14

Please download [gimp-2.99.14-setup.exe](https://download.gimp.org/gimp/v2.99/windows/gimp-2.99.14-setup.exe) and follow below steps to install GIMP.

- Click "Install for all users (recommended)"

    ![](figs/gimp_installation_install_for_all.png)

- Select English and click "OK"

    ![](figs/gimp_installation_select_english.png)

- Click "Continue"

    ![](figs/gimp_installation_click_continue.png)

- Click "Install"

    ![](figs/gimp_installation_install.png)
    ![](figs/gimp_installation_processing.png)

- Click "Finish" to complete the installation of GIMP

    ![](figs/gimp_installation_finish.png)



### Install GIMP AI plugins with OpenVINO™ Backend

 - Open command prompt and download it from Github by git with below instruction.

    ```sh
    cd C:\Users\Public\
    mkdir GIMP
    cd GIMP
    git clone https://github.com/intel/openvino-ai-plugins-gimp.git
    ```
    > __Notes__:
    > This is an example that will create a `GIMP` folder in `C:\Users\Public\`, and then download the package to `C:\Users\Public\GIMP`, you still can define where to place this package by yourself.

- Use same command prompt that used in previous step to include OpenVINO™ environment by below command.

    ```sh
    C:\Users\Public\w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64\setupvars.bat
    ```
    > __Notes__:
    > - Folder name of `w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64` is for OpenVINO™ `2023.2` version and it depends on OpenVINO™ version.
    > - Please ensure the OpenVINO™ environment is set, otherwise, NPU will not work for GIMP AI plugins with OpenVINO™ Backend.

 - Use same command prompt that used in previous steps and follow the command below to install it.

    ```sh
    openvino-ai-plugins-gimp\install.bat
    ```

    This step will take time for downloading necessary packages.

    >**Notes:** This must be run from outside the `openvino-ai-plugins-gimp` directory. Do not run `.\install.bat` from inside of `openvino-ai-plugins-gimp` or other directories, otherwise it could expect to see an error during installation.

- After creating python environment, gimpenv3, and downloaded necessary packages, you will see below log. Please enter "__Y__" to continue setting up the models for all the plugins.

    ![](figs/model_downloading.png)

 - Enter "__Y__" to download openvino stable-diffustion-1.4 model

    ![](figs/model_download_SD1.4.png)

 - Press __\<number\>__ for downloading models you want to run.

    ![](figs/model_downloding%20_SD1.5.png)

 - Download "__1__", SD-1.5 Square (512x512) as an instance. Once installation process is completed when you see the messages below.

    ![](figs/download_SD1.5.png)

    >**Notes:**
    > - The downloaded models include FP16 and INT8 precision, and INT8 precision can be executed on MTL NPU.
    > - Weights is saved at `C:\Users\\<user_name>\openvino-ai-plugins-gimp\weights`.

## Set up GIMP AI plugins with OpenVINO™ Backend on GIMP

Please follow below steps to setup plugin at first time and then you can use GIMP to execute GIMP AI plugins with OpenVINO™ Backend to run Stable-Diffusion or other features.

- Open GIMP application from start menu. (Mandatory step for first launch)

    ![](figs/gimp_launch.png)

- Go to "__Edit__" \> "__Preferences__"

    ![](figs/gimp_preferences.png)

- Scroll down and click "__Folder__" \> "__Plug-ins__" and click ![](figs/add.png) to add the "openvino plugin" path. Then, click "__OK__", then close GIMP application.

    ![](figs/gimp_plugin.png)

    >**Notes:** The Plug-ins path can be found below during run the command "openvino-ai-plugins-gimp\install.bat" in cmd prompt.

    ![](figs/gimp_plugin_path.png)

# Execute Stable-Diffusion in GIMP

>Notes: This section runs `SD-1.5 Square (512x512)` as an example. 

With previous section, the GIMP AI plugins with OpenVINO™ Backend is installed and you can execute stable diffusion in GIMP. This session will guide you to execute Stable-Diffusion in GIMP.

## Execute GIMP

To make GIMP AI plugins with OpenVINO™ Backend can work with GIMP, please choose one of below two methods to execute it.

### Execute by commands in cmd prompt

Open command prompt, then follow below command to execute GIMP in command prompt.

```sh
C:\Users\Public\w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64\setupvars.bat
cd "C:\Program Files\GIMP 2.99\bin"
gimp-2.99.exe
```

### One-click execution by GIMP.bat file

- Create `GIMP.bat` in the directory `C:\Users\Public`.

    ```sh
    call C:\Users\Public\w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64\setupbars.bat
    cd "C:\Program Files\GIMP 2.99\bin"
    .\gimp-2.99.exe 
    ```

 - And then double click `GIMP.bat` or open command prompt executes `GIMP.bat` by below command.

    ```sh
    C:\Users\Public\GIMP.bat
    ```


## Execute Stable-Diffusion – SD1.5_square_int8

Please follow below steps to execute Stable-Diffusion - SD1.5_square_int8. For other features, please refer to [OpenVINO™ Image Generator Plugin with Stable Diffusion](https://github.com/intel/openvino-ai-plugins-gimp/tree/main?tab=readme-ov-file#openvino-image-generator-plugin-with-stable-diffusion) section.

- Following previous section to launch GIMP

    ![](figs/gimp_launch.png)

- Click "__File__" \> "__New__" \> "__OK__" to new a layer in GIMP

    ![](figs/gimp_create_image.png)
    ![](figs/gimp_ok.png)

- Click "__Layer__" \> "__OpenVINO-AI-Plugins__" \> "__Stable diffusion__".

    ![](figs/gimp_execute_SD.png)

- Change the selected fields to set "Stable Diffusion" configuration, then click "Load Models" to the load models into the target devices.

    ![](figs/gimp_load_model.png)

    > **Notes:** It takes time in this step.

- Finally, you can optionally enter any text or changes the parameters in the selected field, then click "Generate" to generate image.

    ![](figs/gimp_sd_ui.png)

    >**Notes:**
    > - Text encoder and VAE can currently run on CPU, or GPU, not NPU.
    > - This demo is based on the following configuration for the most efficient, optimized and good quality image generation:
    >   - Text encoder on CPU
    >   - Unet model runs on GPU and NPU in parallel.
    >   - VAE decoder run on GPU
