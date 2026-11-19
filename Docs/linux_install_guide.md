# Linux Installation Guide

You can install the plugins on Linux in two ways:

* [Install from the Snap Store](#snap-installation) (quickest, for distributions that support snaps)
* [Build GIMP and install the plugins manually](#building-and-installing-the-plugins-manually)

The plugins work with GIMP 3.0 and later 3.x releases; the manual steps below build the latest release, GIMP 3.2.8. The manual steps are tested on Ubuntu 24.04 on x86_64. Flatpak builds of GIMP are not supported.

# Snap Installation

For Linux distributions that support snaps (e.g. Ubuntu), install the plugins and connect them to the GIMP snap with these commands:
```sh
sudo snap install gimp
sudo snap install intel-npu-driver # (optional) for NPU support
sudo snap install openvino-ai-plugins-gimp
sudo snap connect gimp:gimp-plugins openvino-ai-plugins-gimp:gimp-plugins
```
See the [snap README](https://github.com/canonical/openvino-ai-plugins-gimp-snap/blob/main/README.md) for more details, including how models are downloaded.

# Building and installing the plugins manually

The steps below build GIMP 3.2.8 from source. If your distribution already packages GIMP 3.0 or newer (for example, Ubuntu 26.04 ships GIMP 3.2 in `apt`), you can install that instead of building it:
```sh
sudo apt install -y gimp git python3-venv rsync
```
then skip to [Install Plugins](#install-plugins). A packaged GIMP doesn't need the launcher script in [Start GIMP](#start-gimp); start it as usual. The install script detects your GIMP version and copies the plugins into that version's plugin folder, e.g. `~/.config/GIMP/3.2/plug-ins` (see [GIMP versions](#gimp-versions)).

## Build GIMP
1. Create a directory where you will download and build the GIMP sources. The plugin repo and its Python virtual environment will also live here.
    ```sh
    mkdir -p $HOME/GIMP
    cd $HOME/GIMP
    ```
2. Install dependencies
    ```sh
    sudo apt update
    sudo apt install -y cmake libgflags-dev patchelf python3-pip python3-venv python3-gi rsync gettext git git-lfs build-essential meson ninja-build autoconf libcairo2-dev libxt-dev libgdk-pixbuf-2.0-dev libgexiv2-dev libgtk-3-dev libmypaint-dev mypaint-brushes libbz2-dev libfribidi-dev libarchive-dev libatk1.0-dev libgirepository1.0-dev libgirepository-2.0-dev libx11-xcb-dev libwmf-dev libxcb-glx0-dev libxcb-dri2-0-dev libxxf86vm-dev valgrind libappstream-glib-dev libappstream-dev libpugixml-dev libxmu-dev libpoppler-glib-dev xsltproc librsvg2-dev libopencv-dev
    ```
3. Clone, build, and install babl
    ```sh
    git clone https://gitlab.gnome.org/GNOME/babl
    cd babl
    git checkout tags/BABL_0_1_118
    meson setup _build
    ninja -C _build
    sudo ninja -C _build install
    cd ..
    ```
4. Clone, build, and install gegl
    ```sh
    git clone https://gitlab.gnome.org/GNOME/gegl
    cd gegl
    git checkout tags/GEGL_0_4_66
    meson setup _build
    ninja -C _build
    sudo ninja -C _build install
    cd ..
    ```
5. Clone, build, and install GIMP
    ```sh
    git clone https://gitlab.gnome.org/GNOME/gimp
    cd gimp
    git checkout tags/GIMP_3_2_8
    git submodule update --init
    export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0:/usr/local/lib/x86_64-linux-gnu/girepository-1.0
    sudo ldconfig
    meson setup _build
    ninja -C _build
    sudo ninja -C _build install
    cd ..
    ```

> **Note:** The `x86_64-linux-gnu` paths here and below are for 64-bit Intel/AMD systems. On other architectures, use the matching multiarch directory (run `gcc -print-multiarch` to find it).

## GPU and NPU drivers
The plugins always run on the CPU. To also run them on an Intel GPU or NPU, install the drivers OpenVINO needs **before** installing the plugins; the install script records which devices are available. The Linux kernel already includes the GPU and NPU kernel drivers, but OpenVINO also needs these user-space drivers, which are not installed by default.

* **GPU** (Intel Arc and integrated Intel graphics): install the Intel GPU compute runtime (OpenCL and Level Zero):
    ```sh
    sudo apt install -y intel-opencl-icd libze-intel-gpu1 clinfo
    ```
    `clinfo -l` should then list an Intel OpenCL device. On older distributions these packages may be missing, or too old for newer GPUs such as Lunar Lake's Arc 130V/140V; install the latest release from [intel/compute-runtime](https://github.com/intel/compute-runtime/releases) instead.
* **NPU** (Intel Core Ultra): install the Level Zero NPU driver from [intel/linux-npu-driver](https://github.com/intel/linux-npu-driver/releases) and follow the installation steps in its release notes.

Your user must be in the `render` group to use the GPU and NPU. Check with `groups`; if `render` is missing, run `sudo usermod -aG render $USER`, then log out and back in.

After installing the plugins, check which devices OpenVINO can use:
```sh
$HOME/GIMP/gimpenv3/bin/python3 -c "import openvino as ov; print(ov.Core().available_devices)"
```
This should print e.g. `['CPU', 'GPU', 'NPU']`. If you install or update GPU or NPU drivers after installing the plugins, re-run the install script (see [Updating](#updating)) so the plugins pick up the new devices.

## Install Plugins
1. Clone this repo into the same directory
   ```sh
   mkdir -p $HOME/GIMP
   cd $HOME/GIMP
   git clone https://github.com/intel/openvino-ai-plugins-gimp.git
   ```

2. Run the install script **from `$HOME/GIMP`**. It creates the Python virtual environment `$HOME/GIMP/gimpenv3`, installs the required packages and copies the plugins into GIMP's plugin folder. The `-i` option then starts the interactive model downloader so you can pick the Stable Diffusion models to install.
   ```sh
   cd $HOME/GIMP
   chmod +x openvino-ai-plugins-gimp/install.sh
   ./openvino-ai-plugins-gimp/install.sh -i
   ```
   Leave out `-i` to install only the plugins. The script prints the GIMP version it installed the plugins for; to choose it yourself, add `--gimp-version X.Y` (e.g. `--gimp-version 3.0`). The Semantic Segmentation and Super Resolution models are included with the plugins. FastSD downloads its models the first time you use them.

> **Important:** The script creates the virtual environment in the directory you run it from, and the plugins record that location. Always run it from `$HOME/GIMP`, and don't move or rename `gimpenv3` or the repo afterwards. If you do, re-run the install script.

### Downloading more models later
To add Stable Diffusion models at any time, run the model downloader again:
```sh
$HOME/GIMP/gimpenv3/bin/python3 $HOME/GIMP/openvino-ai-plugins-gimp/model_setup.py
```
Models are stored in `~/openvino-ai-plugins-gimp`. To store them somewhere else, set the `GIMP_OPENVINO_MODELS_PATH` environment variable before installing and when starting GIMP.

## Start GIMP
A GIMP built from source in `/usr/local` needs these environment variables **every time it starts**, so put them in a small launcher script:
```sh
cat > $HOME/GIMP/run-gimp.sh << 'EOF'
#!/bin/bash
export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0:/usr/local/lib/x86_64-linux-gnu/girepository-1.0
export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec gimp-3 "$@"
EOF
chmod +x $HOME/GIMP/run-gimp.sh
```
Start GIMP with `$HOME/GIMP/run-gimp.sh`. A GIMP started from the desktop app menu won't have these variables set, unless the menu entry runs this script.

## Verify Installation
Open or create an image. **Layer → OpenVINO-AI-Plugins** should list Semantic Segmentation, Super Resolution, Stable Diffusion and FastSD.

If the menu is missing:
* Check that the plugins are in `~/.config/GIMP/<version>/plug-ins` for your GIMP version, e.g. `~/.config/GIMP/3.2/plug-ins` (folders `semseg_ov`, `superresolution_ov`, `stable_diffusion_ov`, `fastsd_ov` and `openvino_utils`). Run `gimp-3 --version` to see your GIMP version.
* If the install script installed for the wrong version, re-run it with `--gimp-version X.Y`.
* Start GIMP from a terminal with `run-gimp.sh` and look for plugin errors in the output.

If GPU or NPU is missing from a plugin's device list, see [GPU and NPU drivers](#gpu-and-npu-drivers).

## GIMP versions
GIMP reads user plugins from a folder named after its major.minor version, for example `~/.config/GIMP/3.0/plug-ins` for GIMP 3.0.x and `~/.config/GIMP/3.2/plug-ins` for GIMP 3.2.x. The install script finds the installed GIMP version by looking for `gimp-3.x` programs in your `PATH`, `/usr/local/bin` and `/usr/bin`, and picks the newest one. If it can't find GIMP, it assumes 3.2.

To install for a specific version, or for several versions side by side, pass `--gimp-version`:
```sh
cd $HOME/GIMP
./openvino-ai-plugins-gimp/install.sh --gimp-version 3.0
```
When you upgrade GIMP to a new minor version (e.g. 3.0 to 3.2), re-run the install script so the plugins are copied to the new version's folder.

## Updating
```sh
cd $HOME/GIMP/openvino-ai-plugins-gimp
git pull
cd $HOME/GIMP
./openvino-ai-plugins-gimp/install.sh
```
Models you already downloaded are kept.

## Uninstalling
```sh
GIMP_VER=3.2   # the GIMP version the plugins were installed for
rm -rf $HOME/.config/GIMP/$GIMP_VER/plug-ins/{openvino_utils,semseg_ov,stable_diffusion_ov,superresolution_ov,fastsd_ov}
rm -rf $HOME/GIMP/gimpenv3
rm -rf $HOME/openvino-ai-plugins-gimp   # downloaded models
```
