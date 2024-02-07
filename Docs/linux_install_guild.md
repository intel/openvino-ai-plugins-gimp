# Install OpenVINO
1. Download the latest [OpenVINO™](https://github.com/openvinotoolkit/openvino/releases) for your version of Ubuntu.

   https://github.com/gblong1/openvino-ai-plugins-gimp/assets/22227580/79ce7302-4f53-4451-89a2-47d3ef8a29c8


3. Extract the archive
   ```sh
   tar -xvf <ov-release-name>.tar.gz
   ```
4. Create an intel directory under `/opt` and move the extracted files to that directory
   ```sh
   sudo mkdir /opt/intel
   sudo chown <user>:<group> /opt/intel
   mv <ov-release-name> /opt/intel/.
   ```
5. Link the release as `openvino` under `/opt/intel`
   ```sh
   cd /opt/intel
   ln -sf <ov-release-name> openvino
   ```

# Build GIMP
1. Create a directory where you will download and build all of the sources for GIMP
    ```sh
    cd $HOME
    mkdir ./GIMP
    cd ./GIMP
    ```
2. Install Dependencies
    ```sh
    sudo apt update
    sudo apt install -y cmake libgflags-dev patchelf python3-pip gettext git git-lfs build-essential meson ninja-build autoconf libcairo2-dev libxt-dev libgdk-pixbuf-2.0-dev libgexiv2-dev libgtk-3-dev libmypaint-dev mypaint-brushes libbz2-dev libatk1.0-dev libgirepository1.0-dev libx11-xcb-dev libwmf-dev libxcb-glx0-dev  libxcb-dri2-0-dev   libxxf86vm-dev   valgrind  libappstream-glib-dev  libpugixml-dev libxmu-dev   libpoppler-glib-dev   xsltproc librsvg2-dev libopencv-dev
    ```
3. Clone, build, and install babl
    ```sh
    git clone https://gitlab.gnome.org/GNOME/babl
    cd babl
    git checkout tags/BABL_0_1_98
    meson _build
    ninja -C _build
    sudo ninja -C _build install
    cd ..
    ```
4. Clone, build, and install gegl
    ```sh
    git clone https://gitlab.gnome.org/GNOME/gegl
    cd gegl
    git checkout tags/GEGL_0_4_46
    meson _build
    ninja -C _build
    sudo ninja -C _build install
    cd ..
    ```

5. Clone, build, and install Gimp
    ```sh
    git clone https://gitlab.gnome.org/GNOME/gimp  
    cd gimp
    git checkout tags/GIMP_2_99_16
    meson _build
    ninja -C _build
    sudo ninja -C _build install
    cd ..
    ```
# Install Plugins
1. Clone this repo
   ```sh
   cd $HOME/GIMP
   git clone https://github.com/intel/openvino-ai-plugins-gimp.git
   ```

2. Run install script, and download models. The following steps will create the virtual environment "gimpenv3", install all required packages and will also walk you through models setup.
   ```sh
   chmod +x openvino-ai-plugins-gimp/install.sh
   source /opt/intel/openvino/setupvars.sh
   ./openvino-ai-plugins-gimp/install.sh
   ```
   At the end of plugin setup, you will be prompted to setup the AI models used with OpenVINO™. Before proceeding, make note of the plugin directory and add that to the plugin path for GIMP before proceeding with model download.
   Start the GIMP application (`gimp-2.99`), and add the gimpenv3 path that was printed when running the above step to the list of plugin folders  [Edit-> Preferences-> Folders-> Plugins].  

https://github.com/gblong1/openvino-ai-plugins-gimp/assets/22227580/1bab09ae-9383-4d53-9615-eb87d2212e82

   Choose the models that you would like to setup, keeping in mind that choosing to download all of them may take time and considerable disk space.

   *You can re-run "run install script" step later again to install & setup models that you may have missed.*

# Verify Installation
 Start GIMP, ensuring to setup the environment variables correctly,  and you should see 'OpenVINO-AI-Plugins' show up in 'Layer' menu
   ```sh
   source /opt/intel/openvino/setupvars.sh
   export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0:/usr/local/lib/x86_64-linux-gnu/girepository-1.0
   gimp-2.99
   ```

https://github.com/gblong1/openvino-ai-plugins-gimp/assets/22227580/a6222235-2017-45f8-9594-f49ab4107a70

