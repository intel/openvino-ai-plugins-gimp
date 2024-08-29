#!/usr/bin/env python3
# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

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

sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","openvino_utils")])
from plugin_utils import *

_ = gettext.gettext
image_paths = {
    "logo": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "openvino_utils", "images", "plugin_logo.png"
    ),
    "error": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "openvino_utils", "images", "error_icon.png"
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

def check_files_exist(dir_path, files):
    return all(os.path.isfile(Path(dir_path) / file) for file in files)

def list_models(weight_path, SD):
    model_list = []    
    model_paths = {
        "sd_1.4": ["stable-diffusion-ov", "stable-diffusion-1.4"],
        "sd_1.5_square_lcm": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_lcm"],
        "sd_1.5_portrait": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait"],
        "sd_1.5_square": ["stable-diffusion-ov", "stable-diffusion-1.5", "square"],
        "sd_1.5_square_int8": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_int8"],        
        "sd_1.5_square_int8a16" : ["stable-diffusion-ov", "stable-diffusion-1.5", "square_int8"],        
        "sd_1.5_landscape": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape"],
        "sd_1.5_portrait_512x768": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait_512x768"],
        "sd_1.5_landscape_768x512": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape_768x512"],
        "sd_1.5_inpainting": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting"],
        "sd_1.5_inpainting_int8": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting_int8"],
        "sd_2.1_square_base": ["stable-diffusion-ov", "stable-diffusion-2.1", "square_base"],
        "sd_2.1_square": ["stable-diffusion-ov", "stable-diffusion-2.1", "square"],
        "sd_3.0_square": ["stable-diffusion-ov", "stable-diffusion-3.0"],
        "controlnet_referenceonly": ["stable-diffusion-ov", "controlnet-referenceonly"],
        "controlnet_openpose": ["stable-diffusion-ov", "controlnet-openpose"],
        "controlnet_canny": ["stable-diffusion-ov", "controlnet-canny"],
        "controlnet_scribble": ["stable-diffusion-ov", "controlnet-scribble"],
        "controlnet_openpose_int8": ["stable-diffusion-ov", "controlnet-openpose-int8"],
        "controlnet_canny_int8": ["stable-diffusion-ov", "controlnet-canny-int8"],
        "controlnet_scribble_int8": ["stable-diffusion-ov", "controlnet-scribble-int8"],
    }
    # Default path if model_name is not in the dictionary
    dir_path = os.path.join(weight_path, *model_paths.get(SD, ""))
    if Path(dir_path).is_dir():
        model_list.append(SD)
    
    return model_list



    
class SDRunner:
    def __init__ (self, procedure, image, drawable, prompt, negative_prompt, num_images,num_infer_steps, guidance_scale, initial_image,
                  strength, seed, progress_bar, config_path_output):
        self.procedure = procedure
        self.image = image
        self.drawable = drawable
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_images = num_images
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
        num_images = self.num_images
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
                       "num_images": num_images,
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
            cache_image = "sd_cache.png"
            result = Gimp.file_load(
                Gimp.RunMode.NONINTERACTIVE,
                Gio.file_new_for_path(os.path.join(weight_path, "..", cache_image)),
            )
            try:
                # 2.99.10
                result_layer = result.get_active_layer()
            except:
                # > 2.99.10
                result_layers = result.list_layers()
                result_layer = result_layers[0]

            copy = Gimp.Layer.new_from_drawable(result_layer, image_new)
            set_name = "Stable Diffusion -" + str(data_output["seed_num"])
            copy.set_name(set_name)
            copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)  # DIFFERENCE_LEGACY
            image_new.insert_layer(copy, None, -1)

            Gimp.displays_flush()
            image.undo_group_end()
            Gimp.context_pop()
           

            # Remove temporary layers that were saved
            my_dir = os.path.join(weight_path, "..")
            for f_name in os.listdir(my_dir):
                if f_name.startswith("sd_cache"):
                    os.remove(os.path.join(my_dir, f_name))

            self.result = procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
            return self.result

        else:
            show_dialog(
                "Inference not successful. See error_log.txt in GIMP-OpenVINO folder.",
                "Error !",
                "error",
                image_paths
            )
           
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

def async_load_models(python_path, server_path, model_name, supported_devices, device_power_mode,dialog):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        s.sendall(b"kill")

        print("stable-diffusion model server killed")
    except:
        print("No stable-diffusion model server found to kill")

    process = subprocess.Popen([python_path, server_path, model_name, str(supported_devices), device_power_mode], close_fds=True)
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

def async_sd_run_func(runner, dialog,num_images):
    print("Running SD async")
    for i in range(num_images):
        if i != 0:
            runner.seed = None
        runner.run(dialog)
        i += 1
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
            os.path.dirname(os.path.realpath(__file__)), "..", "openvino_utils", "tools"
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
        
        list_layers = []

        try:
            list_layers = image.get_layers()
        except:
            list_layers = image.list_layers()
     
        
        if list_layers[0].get_mask() == None:
            n_layers = 1
            save_image(image, list_layers, os.path.join(config_path_output["weight_path"], "..", "layer_init_image.png"))
        else:
            n_layers = 2
            mask = list_layers[0].get_mask()
            save_image(image, [mask], os.path.join(config_path_output["weight_path"], "..", "cache0.png"))
            save_image(image, list_layers, os.path.join(config_path_output["weight_path"], "..", "cache1.png"))

        if n_layers == 2:
            model_list = (list_models(config_path_output["weight_path"],"sd_1.5_inpainting") +
                          list_models(config_path_output["weight_path"],"sd_1.5_inpainting_int8"))
        else:
            model_list = (list_models(config_path_output["weight_path"],"sd_1.4") +
                          list_models(config_path_output["weight_path"],"sd_1.5_square_lcm") +
                          list_models(config_path_output["weight_path"],"sd_1.5_portrait") +
                          list_models(config_path_output["weight_path"],"sd_1.5_square") +
                          list_models(config_path_output["weight_path"],"sd_1.5_square_int8") +
                          list_models(config_path_output["weight_path"],"sd_1.5_square_int8a16") +
                          list_models(config_path_output["weight_path"],"sd_1.5_landscape") +
                          list_models(config_path_output["weight_path"],"sd_1.5_portrait_512x768") +
                          list_models(config_path_output["weight_path"],"sd_1.5_landscape_768x512") +
                          list_models(config_path_output["weight_path"],"sd_2.1_square_base") +
                          list_models(config_path_output["weight_path"],"sd_2.1_square") +
                          list_models(config_path_output["weight_path"],"sd_3.0_square") +
                          list_models(config_path_output["weight_path"],"controlnet_referenceonly") +
                          list_models(config_path_output["weight_path"],"controlnet_openpose") + 
                          list_models(config_path_output["weight_path"],"controlnet_openpose_int8") +
                          list_models(config_path_output["weight_path"],"controlnet_canny_int8") + 
                          list_models(config_path_output["weight_path"],"controlnet_canny") + 
                          list_models(config_path_output["weight_path"],"controlnet_scribble") + 
                          list_models(config_path_output["weight_path"],"controlnet_scribble_int8"))

        model_name_enum = DeviceEnum(model_list)  
        
        if "NPU" in supported_devices:
            supported_modes = ["Best power efficiency", "Balanced", "Best performance"]
        else:
            supported_modes = ["Best performance"]
        device_name_enum = DeviceEnum(supported_modes)        
        
        config = procedure.create_config()
        config.begin_run(image, run_mode, args)

        # Create JSON Cache - this dictionary will get over witten if the cache exists.
        sd_option_cache_data = dict(prompt="", negative_prompt="", num_images=1,num_infer_steps=20, guidance_scale=7.5,
                                    initial_image=None, strength=0.8, seed="",
                                    inference_status="success", src_height=512, src_width=512)

        sd_option_cache = os.path.join(config_path_output["weight_path"], "..", "gimp_openvino_run_sd.json")

        try:
            with open(sd_option_cache, "r") as file:
                sd_option_cache_data = json.load(file)
               
                # print(json.dumps(sd_option_cache_data, indent=4))
        except:
            print(f"{sd_option_cache} not found, loading defaults")

        GimpUi.init("stable_diffusion_ov.py")
        use_header_bar = Gtk.Settings.get_default().get_property(
            "gtk-dialogs-use-header"
        )
        dialog = GimpUi.Dialog(use_header_bar=use_header_bar, title=_("Stable Diffusion - PLUGIN LICENSE : Apache-2.0"))
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

        #number of images
        num_images_label = Gtk.Label.new_with_mnemonic(_("_Number of Images"))
        num_images_spin = GimpUi.prop_spin_button_new(
            config, "num_images", step_increment=1, page_increment=0.1, digits=0
        ) 

        # num_infer_steps parameter
        steps_label = Gtk.Label.new_with_mnemonic(_("_Number of Inference steps"))
               
        steps_spin = GimpUi.prop_spin_button_new(
            config, "num_infer_steps", step_increment=1, page_increment=0.1, digits=0
        )

        # guidance_scale parameter
        gscale_label = Gtk.Label.new_with_mnemonic(_("_Guidance Scale"))
        
        gscale_spin = GimpUi.prop_spin_button_new(
            config, "guidance_scale", step_increment=0.1, page_increment=0.1, digits=1
        )
        
        

        # seed
        seed = Gtk.Entry.new()
        
        seed.set_width_chars(40)
        seed.set_placeholder_text(_("If left blank, random seed will be set.."))

        seed.set_buffer(Gtk.EntryBuffer.new(sd_option_cache_data["seed"], -1))
        

        seed_text = _("Seed")
        seed_label = Gtk.Label(label=seed_text)


        adv_power_mode_label = Gtk.Label.new_with_mnemonic(_("_Power Mode"))
        adv_power_mode_combo = GimpUi.prop_string_combo_box_new(
            config, "power_mode", device_name_enum.get_tree_model(), 0, 1
        )
        

        adv_checkbox = GimpUi.prop_check_button_new(config, "advanced_setting",
                                                  _("_Advanced Settings                                                       "))
        adv_checkbox.connect("toggled", on_toggled, dialog)
        adv_checkbox.show()
        grid.attach(adv_checkbox, 3, 0, 1, 1)

        invisible_label4 = Gtk.Label.new_with_mnemonic(_("_"))
        invisible_label5 = Gtk.Label.new_with_mnemonic(_("_"))
        invisible_label6 = Gtk.Label.new_with_mnemonic(_("_"))
        invisible_label7 = Gtk.Label.new_with_mnemonic(_("_"))
        invisible_label8 = Gtk.Label.new_with_mnemonic(_("_"))

        grid.attach(invisible_label4,0, 3, 1, 1)
        grid.attach(invisible_label5,0, 4, 1, 1)
        grid.attach(invisible_label6,0, 5, 1, 1)
        grid.attach(invisible_label7,0, 6, 1, 1)
        grid.attach(invisible_label8,0, 7, 1, 1)    
    
        invisible_label4.show()
        invisible_label5.show()
        invisible_label6.show()
        invisible_label7.show()  
        invisible_label8.show()           

        def power_modes_supported(model_name):
            if "sd_1.5_square" in model_name or "int8" in model_name:
                return True
            return False
        
        def remove_all_advanced_widgets():
            grid.remove(gscale_label)
            gscale_label.hide()
            grid.remove(gscale_spin)
            gscale_spin.hide()
           
            grid.remove(steps_label)
            steps_label.hide()
            grid.remove(steps_spin)
            steps_spin.hide()

            grid.remove(num_images_label)
            num_images_label.hide()
            grid.remove(num_images_spin)
            num_images_spin.hide()            
            

            grid.remove(seed)
            seed.hide()

            grid.remove(seed_label)
            seed_label.hide()

            grid.remove(adv_power_mode_label)
            adv_power_mode_label.hide()
            grid.remove(adv_power_mode_combo)
            adv_power_mode_combo.hide()       

            invisible_label4.show()
            invisible_label5.show()
            invisible_label6.show()
            invisible_label7.show()
            invisible_label8.show() 

        def populate_advanced_settings():

            grid.attach(num_images_label, 0, 3, 1, 1)
            grid.attach(num_images_spin, 1, 3, 1, 1)              
           
            grid.attach(steps_label, 0, 4, 1, 1)
            grid.attach(steps_spin, 1, 4, 1, 1)
            grid.attach(gscale_label, 0, 5, 1, 1)
            grid.attach(gscale_spin, 1, 5, 1, 1)
            grid.attach(seed, 1, 6, 1, 1)
            grid.attach(seed_label, 0, 6, 1, 1)
            model_name = config.get_property("model_name")
            
            if power_modes_supported(model_name):
                grid.attach(adv_power_mode_label, 0, 7, 1, 1)
                grid.attach(adv_power_mode_combo, 1, 7, 1, 1)
                adv_power_mode_label.show()
                adv_power_mode_combo.show()



            steps_label.show()
            steps_spin.show()
            gscale_label.show()
            gscale_spin.show()
            seed_label.show()
            seed.show()
            num_images_label.show()
            num_images_spin.show()               
        
            invisible_label4.hide()
            invisible_label5.hide()
            invisible_label6.hide()   
            invisible_label7.hide()
            invisible_label8.hide()           


        if adv_checkbox.get_active():
            populate_advanced_settings()
     

        # Prompt text
        prompt_text = Gtk.Entry.new()
        grid.attach(prompt_text, 1, 1, 1, 1)
        prompt_text.set_width_chars(60)

        prompt_text.set_buffer(Gtk.EntryBuffer.new(sd_option_cache_data["prompt"], -1))
        prompt_text.show()

        prompt_text_label = _("Enter text to generate image")
        prompt_label = Gtk.Label(label=prompt_text_label)
        grid.attach(prompt_label, 0, 1, 1, 1)
       
        prompt_label.show()

        negative_prompt_text_label = _("Negative Prompt")
        negative_prompt_label = Gtk.Label(label=negative_prompt_text_label)
        grid.attach(negative_prompt_label, 0, 2, 1, 1)
  
        negative_prompt_label.show()

        # Negative Prompt text
        negative_prompt_text = Gtk.Entry.new()
        grid.attach(negative_prompt_text, 1, 2, 1, 1)
        negative_prompt_text.set_width_chars(60)
        negative_prompt_text.set_buffer(Gtk.EntryBuffer.new(sd_option_cache_data["negative_prompt"], -1))
        negative_prompt_text.show()


        # UI to browse Initial Image
        def choose_file(widget):
            if file_chooser_dialog.run() == Gtk.ResponseType.OK:
                if file_chooser_dialog.get_file() is not None:
                    # config.set_property("file", file_chooser_dialog.get_file())
                    file_entry.set_text(file_chooser_dialog.get_file().get_path())

            file_chooser_dialog.hide()

        file_chooser_button = Gtk.Button.new_with_mnemonic(label=_("_Init Image from File (Optional)"))
    
        file_chooser_button.connect("clicked", choose_file)

        file_entry = Gtk.Entry.new()
        
      
        file_entry.set_width_chars(40)
        file_entry.set_placeholder_text(_("Choose path..."))
        initial_image = sd_option_cache_data["initial_image"]
        if initial_image is not None:
          
            file_entry.set_text(initial_image) 
    

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
            grid.attach(initialImage_checkbox, 3, 1, 1, 1)
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
        grid.attach_next_to(spinner, sd_run_label, Gtk.PositionType.BOTTOM, 1, 1)

        # Show Logo
        logo = Gtk.Image.new_from_file(image_paths["logo"])
        grid.attach(logo, 3, 3, 2, 3)
        #vbox.pack_start(logo, False, False, 1)
        logo.show()

        # Show License
        #license_text = _("PLUGIN LICENSE : Apache-2.0")
        #label = Gtk.Label(label=license_text)
        #grid.attach(label, 3, 5, 1, 1)
        #vbox.pack_start(label, False, False, 1)
        #label.show()

        progress_bar = Gtk.ProgressBar()
        vbox.add(progress_bar)
        progress_bar.show()

        model_name = config.get_property("model_name")
        device_power_mode = "best performance"

        if model_name == "sd_1.5_square_lcm":
            negative_prompt_label.hide()
            negative_prompt_text.hide()
            initialImage_checkbox.hide()               
        
        if "sd_3.0" in model_name:
            initialImage_checkbox.hide()               

        if is_server_running():
            run_button.set_sensitive(True)
            if adv_checkbox.get_active() and power_modes_supported(model_name): 
                device_power_mode = config.get_property("power_mode")           

        # called when model or device drop down lists are changed.
        # The idea here is that we want to disable the run button
        # if model / devices are changed from what is currently loaded.
        def model_sensitive_combo_changed(widget):
            model_name_tmp = config.get_property("model_name")
            device_power_mode_tmp = None

            # LCM model has no negative prompt
            if model_name_tmp == "sd_1.5_square_lcm":
                negative_prompt_text.hide()
                negative_prompt_label.hide()    
                
            else:
                negative_prompt_text.show()
                negative_prompt_label.show()
                

            if "sd_3.0" in model_name_tmp or "sd_1.5_square_lcm" in model_name_tmp:
                initialImage_checkbox.hide()
            else:
                initialImage_checkbox.show()
                
            if "controlnet" in config.get_property("model_name"):
                
                initialImage_checkbox.set_active(True)
            else:
                initialImage_checkbox.set_active(False)

            if adv_checkbox.get_active():
                if "int8" in model_name:
                    device_power_mode_tmp = config.get_property("power_mode")
                
                else:
                    device_power_mode_tmp = None
    
            if (model_name_tmp==model_name   and
                device_power_mode_tmp==device_power_mode):
                run_button.set_sensitive(True)
            else:
                run_button.set_sensitive(False)

        def model_combo_changed(widget):
            remove_all_advanced_widgets()
            if adv_checkbox.get_active():
                populate_advanced_settings()
      

        model_combo.connect("changed", model_combo_changed)
        model_combo.connect("changed", model_sensitive_combo_changed)
        adv_checkbox.connect("toggled", model_sensitive_combo_changed)
        adv_power_mode_combo.connect("changed", model_sensitive_combo_changed)
 
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
          
                #adv_checkbox.set_sensitive(False)
                prompt = prompt_text.get_text()
                negative_prompt = negative_prompt_text.get_text()
                                  
                if adv_checkbox.get_active():
                    num_images = config.get_property("num_images") 
                    num_infer_steps = config.get_property("num_infer_steps")
                    guidance_scale = config.get_property("guidance_scale")
                    strength = config.get_property("strength")
                    if len(seed.get_text()) != 0:
                        seed = seed.get_text()
                    else:
                        seed = None

                else:
                    num_images = 1
                    num_infer_steps = 20
                    if config.get_property("model_name") == "sd_1.5_square_lcm":
                        num_infer_steps = 4

                    guidance_scale = 7.5
                    seed = None
                    strength = 1.0

                if initialImage_checkbox.get_active() and n_layers == 1:
                    if len(file_entry.get_text()) != 0:
                        initial_image = file_entry.get_text()
                    else:
                        initial_image = os.path.join(config_path_output["weight_path"], "..", "layer_init_image.png")
                else:
                    initial_image = None
                      

                runner = SDRunner(procedure, image, layer, prompt, negative_prompt,num_images, num_infer_steps, guidance_scale, initial_image,
                strength, seed, progress_bar, config_path_output)

                sd_run_label.set_label("Running Stable Diffusion...")
                sd_run_label.show()
                spinner.start()
                spinner.show()

                run_inference_thread = threading.Thread(target=async_sd_run_func, args=(runner, dialog,num_images))
                run_inference_thread.start()
                run_button.set_sensitive(False)
                load_model_button.set_sensitive(False)
                continue
            elif response == Gtk.ResponseType.APPLY:
                model_combo.set_sensitive(False)
 
                #grey-out load & run buttons, show label & start spinner
                load_model_button.set_sensitive(False)
                run_button.set_sensitive(False)
                sd_run_label.set_label("Loading Models...")
                sd_run_label.show()
                spinner.start()
                spinner.show()

                model_name = config.get_property("model_name")

                if power_modes_supported(model_name):
                    if adv_checkbox.get_active():
                        device_power_mode = config.get_property("power_mode")     
                    else:
                        device_power_mode = "Best performance" 

                server = "stable_diffusion_ov_server.py"
                server_path = os.path.join(config_path, server)  

                run_load_model_thread = threading.Thread(target=async_load_models, args=(python_path, server_path, model_name, str(supported_devices), device_power_mode,dialog))
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
                remove_all_advanced_widgets()
                if adv_checkbox.get_active():
                    populate_advanced_settings()
             

            else:
#                dialog.destroy()
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )


class StableDiffusion(Gimp.PlugIn):
    ## Parameters ##
    __gproperties__ = {
        "num_images": (
        int, _("_Number of Images (Default:1)"), "Number of Images to generate", 1, 50, 1,
        GObject.ParamFlags.READWRITE,),        
        "num_infer_steps": (
        int, _("_Number of Inference steps (Default:20)"), "Number of Inference steps (Default:20)", 1, 50, 20,
        GObject.ParamFlags.READWRITE,),
        "guidance_scale": (float, _("_Guidance Scale (Default:7.5)"), "Guidance Scale (Default:7.5)", 1.0001, 20.0, 7.5,
                           GObject.ParamFlags.READWRITE,),
        "strength": (
        float, _("_Strength of Initial Image (Default:0.8)"), "_Strength of Initial Image (Default:0.8)", 0.0, 1.0, 0.8,
        GObject.ParamFlags.READWRITE,),

        "model_name": (
            str,
            _("Model Name"),
            "Model Name: 'sd_1.4', 'sd_1.5'",
            "sd_1.4",
            GObject.ParamFlags.READWRITE,
        ),    

        "advanced_setting": (
            bool,
            _("_Advanced Settings"),
            "Advanced Settings",
            False,
            GObject.ParamFlags.READWRITE,
        ),

        "power_mode": (
            str,
            _("Power Mode"),
            "Power Mode: 'Balanced', 'Best performance'",
            "Best performance",
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
            procedure.set_menu_label(N_("Stable Diffusion..."))
            procedure.set_attribution("Arisha Kumar", "OpenVINO-AI-Plugins", "2023")
            procedure.add_menu_path("<Image>/Layer/OpenVINO-AI-Plugins/")

            # procedure.add_argument_from_property(self, "initial_image")
            procedure.add_argument_from_property(self, "num_images")
            procedure.add_argument_from_property(self, "num_infer_steps")
            procedure.add_argument_from_property(self, "guidance_scale")
            procedure.add_argument_from_property(self, "strength")
            procedure.add_argument_from_property(self, "model_name")
 
            procedure.add_argument_from_property(self, "advanced_setting")
            procedure.add_argument_from_property(self, "power_mode")
           
            procedure.add_argument_from_property(self, "use_initial_image")




        return procedure


Gimp.main(StableDiffusion.__gtype__, sys.argv)
