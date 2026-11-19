#!/bin/bash
MODEL_SETUP=0
DEFAULT_GIMP_VERSION="3.2"

usage() {
    echo "Usage: $0 [-i|--install_models] [--gimp-version X.Y]"
    echo "  -i, --install_models   run the interactive model setup after installing"
    echo "  --gimp-version X.Y     GIMP version to install the plugins for (e.g. 3.0 or 3.2)."
    echo "                         Detected automatically if not given."
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--install_models)
            MODEL_SETUP=1
            shift
            ;;
        --gimp-version)
            if [[ -z "$2" ]]; then
                echo "--gimp-version needs a value, e.g. --gimp-version 3.2"
                exit 1
            fi
            GIMP_VERSION="$2"
            shift 2
            ;;
        --gimp-version=*)
            GIMP_VERSION="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Invalid option: $1"
            usage
            exit 1
            ;;
    esac
done

# Find the major.minor version of the installed GIMP 3.x (e.g. "3.2").
# GIMP installs a binary named gimp-<major.minor>, so look for those first;
# that works even when a source-built GIMP can't run without LD_LIBRARY_PATH.
detect_gimp_version() {
    local candidates=() bin ver
    for bin in $(compgen -c gimp-3. 2>/dev/null) /usr/local/bin/gimp-3.* /usr/bin/gimp-3.*; do
        ver="${bin##*/gimp-}"
        ver="${ver#gimp-}"
        [[ "$ver" =~ ^3\.[0-9]+$ ]] && candidates+=("$ver")
    done
    if [[ ${#candidates[@]} -eq 0 ]]; then
        for bin in gimp-3 gimp; do
            if command -v "$bin" > /dev/null; then
                ver=$(LD_LIBRARY_PATH="/usr/local/lib/$(gcc -print-multiarch 2>/dev/null)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                      "$bin" --version 2>/dev/null | grep -oE '\b3\.[0-9]+' | head -1)
                [[ -n "$ver" ]] && candidates+=("$ver") && break
            fi
        done
    fi
    [[ ${#candidates[@]} -gt 0 ]] && printf '%s\n' "${candidates[@]}" | sort -u -V | tail -1
}

if [[ -z "$GIMP_VERSION" ]]; then
    GIMP_VERSION=$(detect_gimp_version)
    if [[ -z "$GIMP_VERSION" ]]; then
        GIMP_VERSION="$DEFAULT_GIMP_VERSION"
        echo "Could not detect the installed GIMP version, assuming GIMP $GIMP_VERSION."
        echo "Use --gimp-version X.Y to choose a different version."
    else
        echo "Detected GIMP $GIMP_VERSION"
    fi
fi
if [[ ! "$GIMP_VERSION" =~ ^3\.[0-9]+$ ]]; then
    echo "Invalid GIMP version: $GIMP_VERSION (expected major.minor, e.g. 3.2)"
    exit 1
fi
PLUGIN_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/GIMP/$GIMP_VERSION/plug-ins"

# Get the directory of the currently executing script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the current working directory
current_dir="$(pwd)"

# Compare the directories
if [[ "$script_dir" == "$current_dir" ]]; then
    # If they are the same, move up one directory
    cd ..
fi

fail() {
    echo "ERROR: $1" >&2
    echo "**** openvino-ai-plugins-gimp installation FAILED ****" >&2
    exit 1
}

echo "**** openvino-ai-plugins-gimp Setup started ****"

# Create a virtual environment
python3 -m venv gimpenv3 || fail "Could not create the Python virtual environment. On Debian/Ubuntu, install it with: sudo apt install python3-venv"

# Activate the virtual environment
source gimpenv3/bin/activate || fail "Could not activate the virtual environment gimpenv3"

# Upgrade pip and install required packages
pip3 install --upgrade pip || fail "Could not upgrade pip"
pip3 install "$script_dir/." || fail "Could not install the openvino-ai-plugins-gimp Python package"

# post installation steps
python3 -c "from gimpopenvino import install_utils; install_utils.complete_install(repo_weights_dir=r'${script_dir}/weights')" || fail "Post-installation setup failed"

echo "**** openvino-ai-plugins-gimp Setup Ended ****"
# Deactivate the virtual environment
deactivate

# Copy to GIMP plugin dir
echo "Installing plugin in $PLUGIN_DIR"
for d in openvino_utils semseg_ov stable_diffusion_ov superresolution_ov fastsd_ov; do
    mkdir -p "$PLUGIN_DIR/$d"
    rsync -a gimpenv3/lib/python*/site-packages/gimpopenvino/plugins/"$d" "$PLUGIN_DIR/." || fail "Could not copy the $d plugin to $PLUGIN_DIR"
done
echo "*** openvino-ai-plugins-gimp Installed ***"

# If MODEL_SETUP was set, run the model setup
if [[ "$MODEL_SETUP" -eq 1 ]]; then
    echo "**** OpenVINO MODEL SETUP STARTED ****"
    gimpenv3/bin/python3 "$script_dir/model_setup.py" || fail "Model setup failed"
fi

cd "$current_dir"

exit 0
