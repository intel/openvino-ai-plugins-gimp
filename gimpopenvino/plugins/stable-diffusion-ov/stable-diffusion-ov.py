# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

# !/usr/bin/env python3
# coding: utf-8
"""
Perform Stable-diffusion on the current layer.
"""

import gi

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gimp, GimpUi, GObject, GLib, Gio, Gtk
import gettext
import subprocess
import json
# import pickle
import os
import sys
import socket
from enum import IntEnum

import glob
from pathlib import Path

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")])
from plugin_utils import *

_ = gettext.gettext
image_paths = {
    "logo": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "images", "plugin_logo.png"
    ),
    "error": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "images", "error_icon.png"
    ),
}


class StringEnum:
    """
    Helper class for when you want to use strings as keys of an enum. The values would be
    user facing strings that might undergo translation.

    The constructor accepts an even amount of arguments. Each pair of arguments
    is a key/value pair.
    """

    def __init__(self, *args):
        self.keys = []
        self.values = []

        for i in range(len(args) // 2):
            self.keys.append(args[i * 2])
            self.values.append(args[i * 2 + 1])

    def get_tree_model(self):
        """Get a tree model that can be used in GTK widgets."""
        tree_model = Gtk.ListStore(GObject.TYPE_STRING, GObject.TYPE_STRING)
        for i in range(len(self.keys)):
            tree_model.append([self.keys[i], self.values[i]])
        return tree_model


class DeviceEnum:
    def __init__(self, supported_devices):
        self.keys = []
        self.values = []
        for i in supported_devices:
            self.keys.append(i)
            self.values.append(i)


    def get_tree_model(self):
        """Get a tree model that can be used in GTK widgets."""
        tree_model = Gtk.ListStore(GObject.TYPE_STRING, GObject.TYPE_STRING)
        for i in range(len(self.keys)):
            tree_model.append([self.keys[i], self.values[i]])
        return tree_model

    def get_tree_model_no_npu(self):
        """Get a tree model that can be used in GTK widgets."""
        tree_model = Gtk.ListStore(GObject.TYPE_STRING, GObject.TYPE_STRING)
       
        for i in range(len(self.keys)):
            if self.keys[i] != "NPU":
                tree_model.append([self.keys[i], self.values[i]])
        return tree_model

class SDDialogResponse(IntEnum):
    LoadModelComplete = 777
    RunInferenceComplete = 778
    ProgressUpdate = 779

       
def list_models(weight_path, SD):
    model_list = []
    flag = False
    flag_controlnet = False
    flag_referenceonly = False
    if SD == "SD_1.4":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.4")
        flag = True

    if SD == "SD_1.5_Inpainting":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5-inpainting")
        flag = True
    if SD == "controlnet_openpose":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-openpose")
        flag_controlnet = True

    if SD == "controlnet_canny":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-canny")
        flag_controlnet = True

    if SD == "controlnet_scribble":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-scribble")
        flag_controlnet = True

    if SD == "controlnet_referenceonly":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-referenceonly")
        flag_referenceonly = True

    if flag_controlnet:
        text = Path(dir_path) / 'text_encoder.xml'
        unet = Path(dir_path) / 'unet_controlnet.xml'
        vae = Path(dir_path) / 'vae_decoder.xml'
       
        if os.path.isfile(text) and os.path.isfile(unet) and os.path.isfile(vae):
                model_list.append(SD)
        flag_controlnet = False
        
        return model_list
               
    if flag_referenceonly:
        text = Path(dir_path) / 'text_encoder.xml'
        unet_r = Path(dir_path) / 'unet_reference_read.xml'
        unet_w = Path(dir_path) / 'unet_reference_write.xml'
        vae = Path(dir_path) / 'vae_decoder.xml'
        if os.path.isfile(text) and os.path.isfile(unet_r) and os.path.isfile(unet_w) and os.path.isfile(vae):
                model_list.append(SD)
        return model_list

    if flag:
        text = Path(dir_path) / 'text_encoder.xml'
        unet = Path(dir_path) / 'unet.xml'
        vae = Path(dir_path) / 'vae_decoder.xml'
        if os.path.isfile(text) and os.path.isfile(unet) and os.path.isfile(vae):
                model_list.append(SD)
        return model_list

    if SD == "SD_1.5_square_lcm":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5", "square_lcm")
        if os.path.isdir(dir_path):
            model_list.append(SD)
        return model_list  

    if SD ==  "SD_1.5_Inpainting_int8":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5-inpainting-int8")
        if os.path.isdir(dir_path):
            model_list.append(SD)
        return model_list         

    if SD ==  "controlnet_openpose_int8":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-openpose-int8")
        if os.path.isdir(dir_path):
            model_list.append(SD)
        return model_list

    if SD ==  "controlnet_canny_int8":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-canny-int8")
        if os.path.isdir(dir_path):
            model_list.append(SD)
        return model_list        
        
        
    if SD ==  "controlnet_scribble_int8":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "controlnet-scribble-int8")
        if os.path.isdir(dir_path):
            model_list.append(SD)
        return model_list  
        
    if SD == "SD_1.5":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5")

    if SD == "SD_1.5_square_int8":
        dir_path = os.path.join(weight_path, "stable-diffusion-ov", "stable-diffusion-1.5", "square_int8")
        if os.path.isdir(dir_path):
            model_list.append(SD)
        return model_list

    for file in os.scandir(dir_path): #, recursive=True):
        text = Path(file) / 'text_encoder.xml'
        unet = Path(file) / 'unet.xml'
        vae = Path(file) / 'vae_decoder.xml'
        if os.path.isfile(text) and os.path.isfile(vae):
                model = "SD_1.5_" + os.path.basename(file)
                model_list.append(model)

    return model_list   
        


scheduler_name_enum = StringEnum(
    "LMSDiscreteScheduler",
    _("LMSDiscreteScheduler"),
    "PNDMScheduler",
    _("PNDMScheduler"),
    "EulerDiscreteScheduler",
    _("EulerDiscreteScheduler"),
    "DPMSolverMultistepScheduler",
    _("DPMSolverMultistepScheduler"),
    "UniPCMultistepScheduler",
    _("UniPCMultistepScheduler"),
    )
    
    
class SDRunner:
    def __init__ (self, procedure, image, drawable, prompt, negative_prompt, scheduler, num_infer_steps, guidance_scale, initial_image,
                  strength, seed, progress_bar, config_path_output):
        self.procedure = procedure
        self.image = image
        self.drawable = drawable
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.scheduler = scheduler
        self.num_infer_steps = num_infer_steps
        self.guidance_scale = guidance_scale
        self.initial_image = initial_image
        self.strength = strength
        self.seed = seed
        self.progress_bar = progress_bar
        self.config_path_output = config_path_output
        self.result = None

    def run(self, dialog):
        procedure = self.procedure
        image = self.image
        drawable = self.drawable
        prompt = self.prompt
        negative_prompt = self.negative_prompt
        scheduler = self.scheduler
        num_infer_steps = self.num_infer_steps
        guidance_scale = self.guidance_scale
        initial_image = self.initial_image
        strength = self.strength
        seed = self.seed
        progress_bar = self.progress_bar
        config_path_output = self.config_path_output

        # Save inference parameters and layers
        weight_path = config_path_output["weight_path"]
        python_path = config_path_output["python_path"]
        plugin_path = config_path_output["plugin_path"]

        Gimp.context_push()
        image.undo_group_start()

        #save_image(image, drawable, os.path.join(weight_path, "..", "cache.png"))

        # Option Cache
        sd_option_cache = os.path.join(weight_path, "..", "gimp_openvino_run_sd.json")

        with open(sd_option_cache, "w") as file:
            json.dump({"prompt": prompt,
                       "negative_prompt": negative_prompt,
                       "scheduler": scheduler,
                       "num_infer_steps": num_infer_steps,
                       "guidance_scale": guidance_scale,
                       "initial_image": initial_image,
                       "strength": strength,
                       "seed": seed,
                       "inference_status": "started"}, file)

        # Run inference and load as layer
        '''
        subprocess.call([python_path, plugin_path])
        '''
        self.current_step = 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(b"go")
            while True:
                data = s.recv(1024)
                response = data.decode()
                if response == "done":
                    break
                elif response.isdigit():
                    iteration = int(response)
                    self.current_step = iteration
                    dialog.response(SDDialogResponse.ProgressUpdate)

        data_output = {}

        try:
            with open(sd_option_cache, "r") as file:
                data_output = json.load(file)
                # json.dumps(data_output)
        except:
            print(f"ERROR : {sd_option_cache} not found")

        image.undo_group_end()
        Gimp.context_pop()

        if data_output["inference_status"] == "success":
            image_new = Gimp.Image.new(
                data_output["src_width"], data_output["src_height"], 0
            )

            display = Gimp.Display.new(image_new)
            result = Gimp.file_load(
                Gimp.RunMode.NONINTERACTIVE,
                Gio.file_new_for_path(os.path.join(weight_path, "..", "cache.png")),
            )
            try:
                # 2.99.10
                result_layer = result.get_active_layer()
            except:
                # > 2.99.10
                result_layers = result.list_layers()
                result_layer = result_layers[0]

            copy = Gimp.Layer.new_from_drawable(result_layer, image_new)
            copy.set_name("Stable Diffusion")
            copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)  # DIFFERENCE_LEGACY
            image_new.insert_layer(copy, None, -1)

            Gimp.displays_flush()
            image.undo_group_end()
            Gimp.context_pop()

            # Remove temporary layers that were saved
            #my_dir = os.path.join(weight_path, "..")
            #for f_name in os.listdir(my_dir):
            #    if f_name.startswith("cache"):
            #        os.remove(os.path.join(my_dir, f_name))

            self.result = procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
            return self.result

        else:
            show_dialog(
                "Inference not successful. See error_log.txt in GIMP-OpenVINO folder.",
                "Error !",
                "error",
                image_paths
            )
            #os.remove(sd_option_cache)
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

def is_server_running():
    HOST = "127.0.0.1"  # The server's hostname or IP address
    PORT = 65432  # The port used by the server

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(b"ping")
            data = s.recv(1024)
            if data.decode() == "ping":
                return True
    except:
        return False

    return False

def async_load_models(python_path, server_path, device_name, model_name, dialog):

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        s.sendall(b"kill")

        print("stable-diffusion model server killed")
    except:
        print("No stable-diffusion model server found to kill")


    process = subprocess.Popen([python_path, server_path, model_name, device_name[0], device_name[1], device_name[2], device_name[3]], close_fds=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, 65433))
        s.listen()
        while True:
            conn, addr = s.accept()
            with conn:
                while True:
                    data = conn.recv(1024)
                    if data.decode() == "Ready":
                        break
                break

    dialog.response(SDDialogResponse.LoadModelComplete)

def async_sd_run_func(runner, dialog):
    print("Running SD async")
    runner.run(dialog)
    print("async SD done")
    dialog.response(SDDialogResponse.RunInferenceComplete)

def on_toggled(widget, dialog):
    dialog.response(800)

#
# This is what brings up the UI
#
def run(procedure, run_mode, image, n_drawables, layer, args, data):
    if run_mode == Gimp.RunMode.INTERACTIVE:
        # Get all paths
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "tools"
        )

        with open(os.path.join(config_path, "gimp_openvino_config.json"), "r") as file:
            config_path_output = json.load(file)

        python_path = config_path_output["python_path"]
        client = "test-client.py"
        config_path_output["plugin_path"] = os.path.join(config_path, client)
        
        supported_devices = []
        for device in config_path_output["supported_devices"]:
           if 'GNA' not in device:
                supported_devices.append(device)

        device_name_enum = DeviceEnum(supported_devices) #config_path_output["supported_devices"])
        
        list_layers = []

        try:
            list_layers = image.get_layers()
        except:
            list_layers = image.list_layers()
        #n_layers = len(list_layers)
        #print("N LAYERS", n_layers)
        
        if list_layers[0].get_mask() == None:
            n_layers = 1
            save_image(image, list_layers, os.path.join(config_path_output["weight_path"], "..", "layer_init_image.png"))
        else:
            n_layers = 2
            mask = list_layers[0].get_mask()
            save_image(image, [mask], os.path.join(config_path_output["weight_path"], "..", "cache0.png"))
            save_image(image, list_layers, os.path.join(config_path_output["weight_path"], "..", "cache1.png"))

        if n_layers == 2:
            model_list = (list_models(config_path_output["weight_path"],"SD_1.5_Inpainting") +
                          list_models(config_path_output["weight_path"],"SD_1.5_Inpainting_int8"))
        else:
            model_list = (list_models(config_path_output["weight_path"],"SD_1.4") +
                          list_models(config_path_output["weight_path"],"SD_1.5") +
                          list_models(config_path_output["weight_path"],"controlnet_referenceonly") +
                          list_models(config_path_output["weight_path"],"controlnet_openpose") + 
                          list_models(config_path_output["weight_path"],"controlnet_openpose_int8") +
                          list_models(config_path_output["weight_path"],"controlnet_canny_int8") + 
                          list_models(config_path_output["weight_path"],"controlnet_canny") + 
                          list_models(config_path_output["weight_path"],"controlnet_scribble") + 
                          list_models(config_path_output["weight_path"],"controlnet_scribble_int8"))

        model_name_enum = DeviceEnum(model_list)            
        
        config = procedure.create_config()
        config.begin_run(image, run_mode, args)

        # Create JSON Cache - this dictionary will get over witten if the cache exists.
        sd_option_cache_data = dict(prompt="", negative_prompt="", num_infer_steps=20, guidance_scale=7.5,
                                    initial_image=None, strength=0.8, seed="",
                                    inference_status="success", src_height=512, src_width=512,scheduler="")

        sd_option_cache = os.path.join(config_path_output["weight_path"], "..", "gimp_openvino_run_sd.json")

        try:
            with open(sd_option_cache, "r") as file:
                sd_option_cache_data = json.load(file)
               
                # print(json.dumps(sd_option_cache_data, indent=4))
        except:
            print(f"{sd_option_cache} not found, loading defaults")

        GimpUi.init("stable-diffusion-ov.py")
        use_header_bar = Gtk.Settings.get_default().get_property(
            "gtk-dialogs-use-header"
        )
        dialog = GimpUi.Dialog(use_header_bar=use_header_bar, title=_("Stable Diffusion...."))
        dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("_Help", Gtk.ResponseType.HELP)
        load_model_button = dialog.add_button("_Load Models", Gtk.ResponseType.APPLY)
        run_button = dialog.add_button("_Generate", Gtk.ResponseType.OK)
        run_button.set_sensitive(False)

        vbox = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, homogeneous=False, spacing=10
        )
        dialog.get_content_area().add(vbox)
        vbox.show()

        # Create grid to set all the properties inside.
        grid = Gtk.Grid()
        grid.set_column_homogeneous(False)
        grid.set_border_width(10)
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)
        vbox.add(grid)
        grid.show()
        
        # Model Name parameter
        label = Gtk.Label.new_with_mnemonic(_("_Model Name"))
        grid.attach(label, 0, 0, 1, 1)
        label.show()
        model_combo = GimpUi.prop_string_combo_box_new(
            config, "model_name", model_name_enum.get_tree_model(), 0, 1
        )
        grid.attach(model_combo, 1, 0, 1, 1)
        model_combo.show()

        # Device Name parameter
        basic_device_label = Gtk.Label.new_with_mnemonic(_("_Device"))
        basic_device_combo = GimpUi.prop_string_combo_box_new(
            config, "device_name", device_name_enum.get_tree_model_no_npu(), 0, 1
        )

        adv_text_encoder_device_label = Gtk.Label.new_with_mnemonic(_("_Text Device"))
        adv_text_encoder_device_combo = GimpUi.prop_string_combo_box_new(
            config, "text_encode_device_name", device_name_enum.get_tree_model_no_npu(), 0, 1
        )

        adv_unet_encoder_device_label = Gtk.Label.new_with_mnemonic(_("_Unet Device"))
        adv_unet_encoder_combo = GimpUi.prop_string_combo_box_new(
            config, "unet_device_name", device_name_enum.get_tree_model_no_npu(), 0, 1
        )

        adv_vae_decoder_device_label = Gtk.Label.new_with_mnemonic(_("_VAE Device"))
        adv_vae_decoder_device_combo = GimpUi.prop_string_combo_box_new(
            config, "vae_decoder_device_name", device_name_enum.get_tree_model_no_npu(), 0, 1
        )

        adv_unet_positive_device_label = Gtk.Label.new_with_mnemonic(_("_Unet + Device"))
        adv_unet_positive_combo = GimpUi.prop_string_combo_box_new(
            config, "unet_positive_device_name", device_name_enum.get_tree_model(), 0, 1
        )

        adv_unet_negative_device_label = Gtk.Label.new_with_mnemonic(_("_Unet - Device"))
        adv_unet_negative_combo = GimpUi.prop_string_combo_box_new(
            config, "unet_negative_device_name", device_name_enum.get_tree_model(), 0, 1
        )

        adv_checkbox = GimpUi.prop_check_button_new(config, "advanced_device_selection",
                                                  _("_Advanced Device Selection"))
        adv_checkbox.connect("toggled", on_toggled, dialog)
        adv_checkbox.show()

        # Scheduler Name parameter
        scheduler_label = Gtk.Label.new_with_mnemonic(_("_Scheduler"))
        scheduler_combo = GimpUi.prop_string_combo_box_new(
            config, "scheduler", scheduler_name_enum.get_tree_model(), 0, 1
        )
        scheduler_label.show()
        scheduler_combo.show()

        def remove_all_device_widgets():
            grid.remove(basic_device_label)
            basic_device_label.hide()
            grid.remove(basic_device_combo)
            basic_device_combo.hide()
            grid.remove(adv_checkbox)
            grid.remove(scheduler_label)
            grid.remove(scheduler_combo)
            grid.remove(adv_unet_positive_device_label)
            grid.remove(adv_unet_positive_combo)
            grid.remove(adv_unet_negative_device_label)
            grid.remove(adv_unet_negative_combo)
            grid.remove(adv_text_encoder_device_label)
            grid.remove(adv_text_encoder_device_combo)
            grid.remove(adv_unet_encoder_device_label)
            grid.remove(adv_unet_encoder_combo)
            grid.remove(adv_vae_decoder_device_label)
            grid.remove(adv_vae_decoder_device_combo)
            grid.remove(adv_checkbox)
            grid.remove(scheduler_label)
            grid.remove(scheduler_combo)
            adv_text_encoder_device_label.hide()
            adv_text_encoder_device_combo.hide()
            adv_unet_encoder_device_label.hide()
            adv_unet_encoder_combo.hide()
            adv_vae_decoder_device_label.hide()
            adv_vae_decoder_device_combo.hide()
            adv_unet_positive_device_label.hide()
            adv_unet_positive_combo.hide()
            adv_unet_negative_device_label.hide()
            adv_unet_negative_combo.hide()

        def populate_advanced_devices():
            grid.attach(adv_text_encoder_device_label, 2, 0, 1, 1)
            grid.attach(adv_text_encoder_device_combo, 3, 0, 1, 1)
            model_name = config.get_property("model_name")
            if model_name=="SD_1.5_square_int8" or \
               model_name=="controlnet_openpose_int8" or  \
               model_name=="SD_1.5_Inpainting_int8" or  \
               model_name=="controlnet_canny_int8" or  \
               model_name=="controlnet_scribble_int8":
                grid.attach(adv_unet_positive_device_label, 2, 1, 1, 1)
                grid.attach(adv_unet_positive_combo,        3, 1, 1, 1)
                grid.attach(adv_unet_negative_device_label, 2, 2, 1, 1)
                grid.attach(adv_unet_negative_combo,        3, 2, 1, 1)
                grid.attach(adv_vae_decoder_device_label,  2, 3, 1, 1)
                grid.attach(adv_vae_decoder_device_combo,  3, 3, 1, 1)
                grid.attach(adv_checkbox,                  3, 4, 1, 1)
                grid.attach(scheduler_label,               2, 5, 1, 1)
                grid.attach(scheduler_combo,               3, 5, 1, 1)
                adv_text_encoder_device_label.show()
                adv_text_encoder_device_combo.show()
                adv_unet_positive_device_label.show()
                adv_unet_positive_combo.show()
                adv_unet_negative_device_label.show()
                adv_unet_negative_combo.show()
                adv_vae_decoder_device_label.show()
                adv_vae_decoder_device_combo.show()
            else:
                grid.attach(adv_unet_encoder_device_label, 2, 1, 1, 1)
                grid.attach(adv_unet_encoder_combo,        3, 1, 1, 1)
                grid.attach(adv_vae_decoder_device_label,  2, 2, 1, 1)
                grid.attach(adv_vae_decoder_device_combo,  3, 2, 1, 1)
                grid.attach(adv_checkbox,                  3, 3, 1, 1)
                grid.attach(scheduler_label,               2, 4, 1, 1)
                grid.attach(scheduler_combo,               3, 4, 1, 1)
                adv_text_encoder_device_label.show()
                adv_text_encoder_device_combo.show()
                adv_unet_encoder_device_label.show()
                adv_unet_encoder_combo.show()
                adv_vae_decoder_device_label.show()
                adv_vae_decoder_device_combo.show()

        def populate_basic_devices():
            grid.attach(basic_device_label, 2, 0, 1, 1)
            grid.attach(basic_device_combo, 3, 0, 1, 1)
            grid.attach(adv_checkbox,       3, 1, 1, 1)
            grid.attach(scheduler_label,    2, 2, 1, 1)
            grid.attach(scheduler_combo,    3, 2, 1, 1)
            basic_device_label.show()
            basic_device_combo.show()

        if adv_checkbox.get_active():
            populate_advanced_devices()
        else:
            populate_basic_devices()

        # Prompt text
        prompt_text = Gtk.Entry.new()
        grid.attach(prompt_text, 1, 1, 1, 1)
        prompt_text.set_width_chars(60)

        prompt_text.set_buffer(Gtk.EntryBuffer.new(sd_option_cache_data["prompt"], -1))
        prompt_text.show()

        prompt_text_label = _("Enter text to generate image")
        prompt_label = Gtk.Label(label=prompt_text_label)
        grid.attach(prompt_label, 0, 1, 1, 1)
        #vbox.pack_start(prompt_label, False, False, 1)
        prompt_label.show()

        negative_prompt_text_label = _("Negative Prompt")
        negative_prompt_label = Gtk.Label(label=negative_prompt_text_label)
        grid.attach(negative_prompt_label, 0, 2, 1, 1)
        #vbox.pack_start(negative_prompt_label, False, False, 1)
        negative_prompt_label.show()

        # Negative Prompt text
        negative_prompt_text = Gtk.Entry.new()
        grid.attach(negative_prompt_text, 1, 2, 1, 1)
        negative_prompt_text.set_width_chars(60)
        negative_prompt_text.set_buffer(Gtk.EntryBuffer.new(sd_option_cache_data["negative_prompt"], -1))
        negative_prompt_text.show()

        # num_infer_steps parameter
        label = Gtk.Label.new_with_mnemonic(_("_Number of Inference steps"))
        grid.attach(label, 0, 3, 1, 1)
        label.show()
        spin = GimpUi.prop_spin_button_new(
            config, "num_infer_steps", step_increment=1, page_increment=0.1, digits=0
        )
        grid.attach(spin, 1, 3, 1, 1)
        spin.show()

        # guidance_scale parameter
        label = Gtk.Label.new_with_mnemonic(_("_Guidance Scale"))
        grid.attach(label, 0, 4, 1, 1)
        label.show()
        spin = GimpUi.prop_spin_button_new(
            config, "guidance_scale", step_increment=0.1, page_increment=0.1, digits=1
        )
        grid.attach(spin, 1, 4, 1, 1)
        spin.show()

        # seed
        seed = Gtk.Entry.new()
        grid.attach(seed, 1, 5, 1, 1)
        seed.set_width_chars(40)
        seed.set_placeholder_text(_("If left blank, random seed will be set.."))

        seed.set_buffer(Gtk.EntryBuffer.new(sd_option_cache_data["seed"], -1))
        seed.show()

        seed_text = _("Seed")
        seed_label = Gtk.Label(label=seed_text)
        grid.attach(seed_label, 0, 5, 1, 1)
        #vbox.pack_start(seed_label, False, False, 1)
        seed_label.show()


        grid.attach(spin, 1, 6, 1, 1)
        spin.show()


        # UI to browse Initial Image
        def choose_file(widget):
            if file_chooser_dialog.run() == Gtk.ResponseType.OK:
                if file_chooser_dialog.get_file() is not None:
                    # config.set_property("file", file_chooser_dialog.get_file())
                    file_entry.set_text(file_chooser_dialog.get_file().get_path())

            file_chooser_dialog.hide()

        file_chooser_button = Gtk.Button.new_with_mnemonic(label=_("_Init Image from File (Optional)"))
        #grid.attach(file_chooser_button, 0, 7, 1, 1)
        #file_chooser_button.show()
        file_chooser_button.connect("clicked", choose_file)

        file_entry = Gtk.Entry.new()
        
        #grid.attach(file_entry, 1, 7, 1, 1)
        file_entry.set_width_chars(40)
        file_entry.set_placeholder_text(_("Choose path..."))
        initial_image = sd_option_cache_data["initial_image"]
        if initial_image is not None:
            # print("initial_image",initial_image)
            file_entry.set_text(initial_image) #.get_path())
        #file_entry.show()

        file_chooser_dialog = Gtk.FileChooserDialog(
            use_header_bar=use_header_bar,
            title=_("Initial Image path..."),
            action=Gtk.FileChooserAction.OPEN,
        )
        file_chooser_dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        file_chooser_dialog.add_button("_OK", Gtk.ResponseType.OK)
        
        # Initial_image strength parameter
        strength_label = Gtk.Label.new_with_mnemonic(_("_Strength of Initial Image"))
        invisible_label1 = Gtk.Label.new_with_mnemonic(_("_"))
        invisible_label2 = Gtk.Label.new_with_mnemonic(_("_"))
        invisible_label3 = Gtk.Label.new_with_mnemonic(_("_"))
        
        spin = GimpUi.prop_spin_button_new(
            config, "strength", step_increment=0.1, page_increment=0.1, digits=1
        )

        initialImage_checkbox = GimpUi.prop_check_button_new(config, "use_initial_image", 
                                    _("_Use Inital Image (Default: Selected layer in Canvas)"))   

        def populate_init_image_section():
            grid.attach(file_chooser_button, 0, 8, 1, 1)
            file_chooser_button.show()
            grid.attach(file_entry, 1, 8, 1, 1)
            file_entry.show()
            grid.attach(strength_label, 0, 9, 1, 1)
            strength_label.show()
            grid.attach(spin, 1, 9, 1, 1)
            spin.show()
            
        if n_layers == 1:
            grid.attach(initialImage_checkbox, 1, 7, 1, 1)
            initialImage_checkbox.show()
            grid.attach(invisible_label1,1, 8, 1, 1)
            grid.attach(invisible_label2,1, 9, 1, 1)
            #grid.attach(invisible_label3,0, 10, 1, 1)
            invisible_label1.show()
            invisible_label2.show()
            #invisible_label3.show()            
            if initialImage_checkbox.get_active():
                populate_init_image_section()     
        
        def initImage_toggled(widget):
            if initialImage_checkbox.get_active():
                invisible_label1.hide()
                invisible_label2.hide()
                invisible_label3.hide()
                populate_init_image_section()
            else:
                file_chooser_button.hide()
                file_entry.hide()
                strength_label.hide()
                spin.hide()
                grid.attach(invisible_label1,1, 8, 1, 1)
                grid.attach(invisible_label2,1, 9, 1, 1)
                #grid.attach(invisible_label3,0, 10, 1, 1)
                invisible_label1.show()
                invisible_label2.show()
                #invisible_label3.show()            

        initialImage_checkbox.connect("toggled", initImage_toggled)    

        # status label
        sd_run_label = Gtk.Label(label="Running Stable Diffusion...") 
        grid.attach(sd_run_label, 1, 12, 1, 1)

        # spinner
        spinner = Gtk.Spinner()
        grid.attach_next_to(spinner, sd_run_label, Gtk.PositionType.RIGHT, 1, 1)

        # Show Logo
        logo = Gtk.Image.new_from_file(image_paths["logo"])
        # grid.attach(logo, 0, 0, 1, 1)
        vbox.pack_start(logo, False, False, 1)
        logo.show()

        # Show License
        license_text = _("PLUGIN LICENSE : Apache-2.0")
        label = Gtk.Label(label=license_text)
        # grid.attach(label, 1, 1, 1, 1)
        vbox.pack_start(label, False, False, 1)
        label.show()

        progress_bar = Gtk.ProgressBar()
        vbox.add(progress_bar)
        progress_bar.show()


        device_text = None
        device_pos_unet = None
        device_neg_unet = None
        device_vae = None
        model_name = config.get_property("model_name")

        if model_name == "SD_1.5_square_lcm":
            negative_prompt_label.hide()
            negative_prompt_text.hide()
            scheduler_label.hide()
            scheduler_combo.hide()


        if is_server_running():
            run_button.set_sensitive(True)
            if adv_checkbox.get_active():
                device_text = config.get_property("text_encode_device_name")
                if model_name=="SD_1.5_square_int8" or model_name=="controlnet_openpose_int8" or  model_name=="SD_1.5_Inpainting_int8" or model_name=="controlnet_canny_int8" or  model_name=="controlnet_scribble_int8":
                    device_pos_unet = config.get_property("unet_positive_device_name")
                    device_neg_unet = config.get_property("unet_negative_device_name")
                else:
                    device_pos_unet = config.get_property("unet_device_name")
                    device_neg_unet = device_pos_unet
                device_vae = config.get_property("vae_decoder_device_name")
            else:
                device = config.get_property("device_name")
                device_text = device
                device_pos_unet = device
                device_neg_unet = device
                device_vae = device


        # called when model or device drop down lists are changed.
        # The idea here is that we want to disable the run button
        # if model / devices are changed from what is currently loaded.
        def model_sensitive_combo_changed(widget):
            model_name_tmp = config.get_property("model_name")

            # LCM model has no negative prompt
            if model_name_tmp == "SD_1.5_square_lcm":
                negative_prompt_text.hide()
                negative_prompt_label.hide()
                scheduler_label.hide()
                scheduler_combo.hide()
            else:
                negative_prompt_text.show()
                negative_prompt_label.show()
                scheduler_label.show()
                scheduler_combo.show()

            if adv_checkbox.get_active():
                device_text_tmp = config.get_property("text_encode_device_name")
                if (model_name=="SD_1.5_square_int8"       or
                    model_name=="controlnet_openpose_int8" or
                    model_name=="SD_1.5_Inpainting_int8"   or
                    model_name=="controlnet_canny_int8"    or
                    model_name=="controlnet_scribble_int8"):
                    device_pos_unet_tmp = config.get_property("unet_positive_device_name")
                    device_neg_unet_tmp = config.get_property("unet_negative_device_name")
                else:
                    device_pos_unet_tmp = config.get_property("unet_device_name")
                    device_neg_unet_tmp = device_pos_unet_tmp
                device_vae_tmp = config.get_property("vae_decoder_device_name")
            else:
                device_tmp = config.get_property("device_name")
                device_text_tmp = device_tmp
                device_pos_unet_tmp = device_tmp
                device_neg_unet_tmp = device_tmp
                device_vae_tmp = device_tmp
            if (model_name_tmp==model_name           and
                device_text_tmp==device_text         and
                device_pos_unet_tmp==device_pos_unet and
                device_neg_unet_tmp==device_neg_unet and
                device_vae_tmp==device_vae):
                run_button.set_sensitive(True)
            else:
                run_button.set_sensitive(False)

        def model_combo_changed(widget):
            remove_all_device_widgets()
            if adv_checkbox.get_active():
                populate_advanced_devices()
            else:
                populate_basic_devices()


        model_combo.connect("changed", model_combo_changed)
        model_combo.connect("changed", model_sensitive_combo_changed)
        basic_device_combo.connect("changed", model_sensitive_combo_changed)
        adv_text_encoder_device_combo.connect("changed", model_sensitive_combo_changed)
        adv_unet_encoder_combo.connect("changed", model_sensitive_combo_changed)
        adv_vae_decoder_device_combo.connect("changed", model_sensitive_combo_changed)
        adv_checkbox.connect("toggled", model_sensitive_combo_changed)

        adv_unet_positive_combo.connect("changed", model_sensitive_combo_changed)
        adv_unet_negative_combo.connect("changed", model_sensitive_combo_changed)

        # Wait for user to click
        dialog.show()
     
        import threading
        run_inference_thread = None
        run_load_model_thread = None

        runner = None
        
        while True:
            response = dialog.run()                           
                
            if response == Gtk.ResponseType.OK:

                model_combo.set_sensitive(False)
                basic_device_combo.set_sensitive(False)
                adv_text_encoder_device_combo.set_sensitive(False)
                adv_unet_encoder_combo.set_sensitive(False)
                adv_vae_decoder_device_combo.set_sensitive(False)
                adv_checkbox.set_sensitive(False)
                adv_unet_positive_combo.set_sensitive(False)
                adv_unet_negative_combo.set_sensitive(False)

                prompt = prompt_text.get_text()
                negative_prompt = negative_prompt_text.get_text()
                scheduler = config.get_property("scheduler")
                num_infer_steps = config.get_property("num_infer_steps")
                guidance_scale = config.get_property("guidance_scale")
                strength = config.get_property("strength")


                if initialImage_checkbox.get_active() and n_layers == 1:
                    if len(file_entry.get_text()) != 0:
                        initial_image = file_entry.get_text()
                    else:
                        initial_image = os.path.join(config_path_output["weight_path"], "..", "layer_init_image.png")
                else:
                    initial_image = None
                    
                if len(seed.get_text()) != 0:
                    seed = seed.get_text()
                else:
                    seed = None
           

                runner = SDRunner(procedure, image, layer, prompt, negative_prompt,scheduler, num_infer_steps, guidance_scale, initial_image,
                strength, seed, progress_bar, config_path_output)

                sd_run_label.set_label("Running Stable Diffusion...")
                sd_run_label.show()
                spinner.start()
                spinner.show()

                run_inference_thread = threading.Thread(target=async_sd_run_func, args=(runner, dialog))
                run_inference_thread.start()
                run_button.set_sensitive(False)
                load_model_button.set_sensitive(False)
                continue
            elif response == Gtk.ResponseType.APPLY:
                model_combo.set_sensitive(False)
                basic_device_combo.set_sensitive(False)
                adv_text_encoder_device_combo.set_sensitive(False)
                adv_unet_encoder_combo.set_sensitive(False)
                adv_vae_decoder_device_combo.set_sensitive(False)
                adv_checkbox.set_sensitive(False)
                adv_unet_positive_combo.set_sensitive(False)
                adv_unet_negative_combo.set_sensitive(False)

                #grey-out load & run buttons, show label & start spinner
                load_model_button.set_sensitive(False)
                run_button.set_sensitive(False)
                sd_run_label.set_label("Loading Models...")
                sd_run_label.show()
                spinner.start()
                spinner.show()

                model_name = config.get_property("model_name")

                if adv_checkbox.get_active():
                    device_text = config.get_property("text_encode_device_name")
                    if model_name=="SD_1.5_square_int8" or model_name=="controlnet_openpose_int8" or  model_name=="SD_1.5_Inpainting_int8" or model_name=="controlnet_canny_int8" or  model_name=="controlnet_scribble_int8":
                        device_pos_unet = config.get_property("unet_positive_device_name")
                        device_neg_unet = config.get_property("unet_negative_device_name")
                    else:
                        device_pos_unet = config.get_property("unet_device_name")
                        device_neg_unet = device_pos_unet
                    device_vae = config.get_property("vae_decoder_device_name")
                    device_name = [device_text, device_pos_unet, device_neg_unet, device_vae]
                else:
                    device = config.get_property("device_name")
                    device_text = device
                    device_pos_unet = device
                    device_neg_unet = device
                    device_vae = device
                    device_name = [device, device, device, device]

                server = "stable-diffusion-ov-server.py"
                server_path = os.path.join(config_path, server)
                
                #def async_load_models(python_path, server_path, device_name, model_name, label, spinner, run_button, load_model_button):
                run_load_model_thread = threading.Thread(target=async_load_models, args=(python_path, server_path, device_name, model_name, dialog))
                run_load_model_thread.start()

                continue
            elif response == Gtk.ResponseType.HELP:
                url = "https://github.com/intel/openvino-ai-plugins-gimp.git"
                Gio.app_info_launch_default_for_uri(url, None)
                continue
            elif response == SDDialogResponse.LoadModelComplete:
                print("model load complete.")
                if run_load_model_thread:
                    run_load_model_thread.join()
                    run_load_model_thread = None

                #re-enable load & run buttons, hide label & stop spinner
                spinner.stop()
                spinner.hide()
                sd_run_label.hide()
                run_button.set_sensitive(True)
                load_model_button.set_sensitive(True)
                model_combo.set_sensitive(True)
                basic_device_combo.set_sensitive(True)
                adv_text_encoder_device_combo.set_sensitive(True)
                adv_unet_encoder_combo.set_sensitive(True)
                adv_vae_decoder_device_combo.set_sensitive(True)
                adv_checkbox.set_sensitive(True)
                adv_unet_positive_combo.set_sensitive(True)
                adv_unet_negative_combo.set_sensitive(True)
            elif response == SDDialogResponse.RunInferenceComplete:
                print("run inference complete.")
                if run_inference_thread:
                    run_inference_thread.join()
                    result = runner.result
                    if result.index(0) == Gimp.PDBStatusType.SUCCESS and config is not None:
                        config.end_run(Gimp.PDBStatusType.SUCCESS)
                    return result
            elif response == SDDialogResponse.ProgressUpdate:
                progress_string=""
                if runner.current_step == runner.num_infer_steps:
                    progress_string = "Running Stable Diffusion... Finalizing Generated Image"
                else:
                    progress_string = "Running Stable Diffusion... (Inference Step " + str(runner.current_step + 1) +  " / " + str(runner.num_infer_steps) + ")"

                sd_run_label.set_label(progress_string)
                if runner.num_infer_steps > 0:
                    perc_complete = runner.current_step / runner.num_infer_steps
                    progress_bar.set_fraction(perc_complete)
            elif response == 800:
                remove_all_device_widgets()
                if adv_checkbox.get_active():
                    populate_advanced_devices()
                else:
                    populate_basic_devices()

            else:
#                dialog.destroy()
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )


class StableDiffusion(Gimp.PlugIn):
    ## Parameters ##
    __gproperties__ = {
        "num_infer_steps": (
        int, _("_Number of Inference steps (Default:32)"), "Number of Inference steps (Default:32)", 1, 50, 32,
        GObject.ParamFlags.READWRITE,),
        "guidance_scale": (float, _("_Guidance Scale (Default:7.5)"), "_Guidance Scale (Default:7.5)", 1.0, 20.0, 7.5,
                           GObject.ParamFlags.READWRITE,),
        "strength": (
        float, _("_Strength of Initial Image (Default:0.8)"), "_Strength of Initial Image (Default:0.8)", 0.0, 1.0, 0.8,
        GObject.ParamFlags.READWRITE,),
        "device_name": (
            str,
            _("Device Name"),
            "Device Name: 'CPU', 'GPU'",
            "CPU",
            GObject.ParamFlags.READWRITE,
        ),
        "model_name": (
            str,
            _("Model Name"),
            "Model Name: 'SD_1.4', 'SD_1.5'",
            "SD_1.4",
            GObject.ParamFlags.READWRITE,
        ),    
        "scheduler": (
            str,
            _("Scheduler"),
            "Scheduler Name: 'LMSDiscreteScheduler', 'PNDMScheduler','EulerDiscreteScheduler'",
            "LMSDiscreteScheduler",
            GObject.ParamFlags.READWRITE,
        ),
        "text_encode_device_name": (
            str,
            _("Text Encoder Device Name"),
            "Device Name: 'CPU', 'GPU'",
            "CPU",
            GObject.ParamFlags.READWRITE,
        ),
        "unet_device_name": (
            str,
            _("Unet Device Name"),
            "Device Name: 'CPU', 'GPU'",
            "CPU",
            GObject.ParamFlags.READWRITE,
        ),
        "vae_decoder_device_name": (
            str,
            _("VAE Decoder Device Name"),
            "Device Name: 'CPU', 'GPU'",
            "CPU",
            GObject.ParamFlags.READWRITE,
        ),
        "advanced_device_selection": (
            bool,
            _("_Advanced Device Selection"),
            "Advanced Device Selection",
            False,
            GObject.ParamFlags.READWRITE,
        ),
        "unet_positive_device_name": (
            str,
            _("Unet Device Name"),
            "Device Name: 'CPU', 'GPU'",
            "CPU",
            GObject.ParamFlags.READWRITE,
        ),
        "unet_negative_device_name": (
            str,
            _("Unet Device Name"),
            "Device Name: 'CPU', 'GPU'",
            "CPU",
            GObject.ParamFlags.READWRITE,
        ),
        
        "use_initial_image": (
            bool,
            _("_Use Initial Image (Default: Open Image in Canvas"),
            "Use Initial Image (Default: Open Image in Canvas",
            False,
            GObject.ParamFlags.READWRITE,
        ),        


        # "initial_image": (str, _("_Init Image (Optional)..."), "_Init Image (Optional)...", None, GObject.ParamFlags.READWRITE,),

    }

    ## GimpPlugIn virtual methods ##
    def do_query_procedures(self):

        try:
            self.set_translation_domain(
                "gimp30-python", Gio.file_new_for_path(Gimp.locale_directory())
            )
        except:
            print("Error in set_translation_domain. This is expected if running GIMP 2.99.11 or later")

        return ["stable-diffusion-ov"]

    def do_set_i18n(self, procname):
        return True, 'gimp30-python', None

    def do_create_procedure(self, name):
        procedure = None
        if name == "stable-diffusion-ov":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, run, None
            )
            procedure.set_image_types("*")
            procedure.set_documentation(
                N_("stable-diffusion on the current layer."),
                globals()[
                    "__doc__"
                ],  # This includes the docstring, on the top of the file
                name,
            )
            procedure.set_menu_label(N_("Stable diffusion..."))
            procedure.set_attribution("Arisha Kumar", "OpenVINO-AI-Plugins", "2023")
            procedure.add_menu_path("<Image>/Layer/OpenVINO-AI-Plugins/")

            # procedure.add_argument_from_property(self, "initial_image")
            procedure.add_argument_from_property(self, "num_infer_steps")
            procedure.add_argument_from_property(self, "guidance_scale")
            procedure.add_argument_from_property(self, "strength")
            procedure.add_argument_from_property(self, "model_name")
            procedure.add_argument_from_property(self, "scheduler") 
            procedure.add_argument_from_property(self, "device_name")
            procedure.add_argument_from_property(self, "text_encode_device_name")
            procedure.add_argument_from_property(self, "unet_device_name")
            procedure.add_argument_from_property(self, "vae_decoder_device_name")
            procedure.add_argument_from_property(self, "advanced_device_selection")
            procedure.add_argument_from_property(self, "unet_positive_device_name")
            procedure.add_argument_from_property(self, "unet_negative_device_name")
            procedure.add_argument_from_property(self, "use_initial_image")




        return procedure


Gimp.main(StableDiffusion.__gtype__, sys.argv)
