#!/usr/bin/env python3
# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

# coding: utf-8
"""
Perform superresolution on the current layer.
"""
import gi
import gettext
import subprocess
import json
import os
import sys
from enum import IntEnum

sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","openvino_utils")])
from plugin_utils import *

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gimp, GimpUi, GObject, GLib, Gio, Gtk

_ = gettext.gettext

from tools.tools_utils import base_model_dir, config_path_dir

image_paths = {
    "logo": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "openvino_utils", "images", "plugin_logo.png"
    ),
    "error": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "openvino_utils", "images", "error_icon.png"
    ),
}


def N_(message): return message
def _(message): return GLib.dgettext(None, message)

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
        self.keys = supported_devices
        self.values = supported_devices

    def get_tree_model(self):
        """Get a tree model that can be used in GTK widgets."""
        tree_model = Gtk.ListStore(GObject.TYPE_STRING, GObject.TYPE_STRING)
        for i in range(len(self.keys)):
            tree_model.append([self.keys[i], self.values[i]])
        return tree_model

model_name_enum = StringEnum(
    "esrgan", _("esrgan"),
    "sr_1033", _("sr_1033"),
    "sr_1032", _("sr_1032"),
)
class SRDialogResponse(IntEnum):
    RunInferenceComplete = 778
    ProgressUpdate = 779

def async_run_superes(runner, dialog):
    print("Running SR async")
    runner.run(dialog)
    print("async SR done")
    dialog.response(SRDialogResponse.RunInferenceComplete)

class SRRunner:
    def __init__ (self, procedure, image, drawable,scale, device_name, model_name, progress_bar, config_path_output):
        self.procedure = procedure
        self.image = image
        self.drawable = drawable
        self.scale = scale
        self.device_name = device_name
        self.model_name = model_name
        self.progress_bar = progress_bar
        self.config_path_output = config_path_output
        self.result = None

    def load_inference_results(self, weight_path):
        with open(os.path.join(weight_path, "..", "gimp_openvino_run.json"), "r") as file:
            return json.load(file)
    
    def run(self, diaglog):
        procedure = self.procedure
        image = self.image
        drawable = self.drawable
        scale = self.scale
        model_name = self.model_name
        device_name = self.device_name
        progress_bar = self.progress_bar
        config_path_output = self.config_path_output
        # Save inference parameters and layers
        weight_path = config_path_output["weight_path"]
        python_path = config_path_output["python_path"]
        plugin_path = config_path_output["plugin_path"]

        Gimp.context_push()
        image.undo_group_start()

        save_image(image, drawable, os.path.join(weight_path, "..", "cache.png"))
        save_inference_parameters(weight_path, device_name, scale, model_name)

        try:
            if sys.platform == 'win32':
                creationflags = subprocess.CREATE_NO_WINDOW 
            else:
                creationflags = 0 # N/A on linux 
  
            subprocess.call([python_path, plugin_path],
                        creationflags=creationflags, 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        )
            data_output = self.load_inference_results(weight_path)
        except Exception as e:
            Gimp.message(f"Error during inference: {e}")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        image.undo_group_end()
        Gimp.context_pop()

        if data_output["inference_status"] == "success":
            try:
                result_layer = handle_successful_inference(weight_path, image, drawable, scale)
            except Exception as e:
                Gimp.message(f"Error processing inference results: {e}")
                return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
    
            Gimp.displays_flush()
            remove_temporary_files(os.path.join(weight_path, ".."))
            self.result = procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
        else:
            show_dialog(
            "Inference not successful. See error_log.txt in GIMP-OpenVINO folder.",
            "Error !",
            "error",
            image_paths
        )
        self.result = procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

def handle_successful_inference(weight_path, image, drawable, scale):
    if scale == 1:
        result = Gimp.file_load(
            Gimp.RunMode.NONINTERACTIVE,
            Gio.file_new_for_path(os.path.join(weight_path, "..", "cache.png")),
        )
        result_layer = result.get_active_layer()
        copy = Gimp.Layer.new_from_drawable(result_layer, image)
        copy.set_name("Super Resolution")
        copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)
        image.insert_layer(copy, None, -1)
    else:
        image_new = Gimp.Image.new(
            drawable[0].get_width() * scale, drawable[0].get_height() * scale, 0
        )
        display = Gimp.Display.new(image_new)
        result = Gimp.file_load(
            Gimp.RunMode.NONINTERACTIVE,
            Gio.File.new_for_path(os.path.join(weight_path, "..", "cache.png")),
        )

        result_layer = result.get_layers()[0]
        copy = Gimp.Layer.new_from_drawable(result_layer, image_new)
        copy.set_name("Super Resolution")
        copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)
        image_new.insert_layer(copy, None, -1)
    return result_layer

def save_inference_parameters(weight_path, device_name, scale, model_name):
    parameters = {
        "device_name": device_name,
        "scale": float(scale),
        "model_name": model_name,
        "inference_status": "started"
    }
    with open(os.path.join(weight_path, "..", "gimp_openvino_run.json"), "w") as file:
        json.dump(parameters, file)


def remove_temporary_files(directory):
    for f_name in os.listdir(directory):
        if f_name.startswith("cache"):
            os.remove(os.path.join(directory, f_name))

# this is what brings up the UI
def run(procedure, run_mode, image, layer, config, data):
    scale = config.get_property("scale") 
    device_name = config.get_property("device_name") 
    model_name = config.get_property("model_name") 
    
    if run_mode == Gimp.RunMode.INTERACTIVE:
        with open(os.path.join(config_path_dir, "gimp_openvino_config.json"), "r") as file:
            config_path_output = json.load(file)
        
        plugin_version = config_path_output["plugin_version"]
        config_path_output["plugin_path"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 
            "..", 
            "openvino_utils", 
            "tools", 
            "superresolution_ov.py")
        
        device_name_enum = DeviceEnum(config_path_output["supported_devices"])

        config = procedure.create_config()
        
        GimpUi.init("superresolution-ov")
        use_header_bar = Gtk.Settings.get_default().get_property("gtk-dialogs-use-header")
        title_bar_label =  "Super Resolution : " +  plugin_version

        dialog = GimpUi.Dialog(use_header_bar=use_header_bar, title=_(title_bar_label))
        dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("_Help", Gtk.ResponseType.APPLY)
        run_button = dialog.add_button("_Generate", Gtk.ResponseType.OK)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, homogeneous=False, spacing=10)
        dialog.get_content_area().add(vbox)
        vbox.show()

        grid = Gtk.Grid()
        grid.set_column_homogeneous(False)
        grid.set_border_width(10)
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)
        vbox.add(grid)
        grid.show()

       # Scale parameter
        label = Gtk.Label.new_with_mnemonic(_("_Scale"))
        grid.attach(label, 0, 2, 1, 1)
        label.show()
        spin = GimpUi.prop_spin_button_new(
            config, "scale", step_increment=1, page_increment=0.1, digits=2
        )
        grid.attach(spin, 1, 2, 1, 1)
        spin.show()

        # Model Name parameter
        label = Gtk.Label.new_with_mnemonic(_("_Model Name"))
        grid.attach(label, 0, 0, 1, 1)
        label.show()
        combo = GimpUi.prop_string_combo_box_new(
            config, "model_name", model_name_enum.get_tree_model(), 0, 1
        )
        grid.attach(combo, 1, 0, 1, 1)
        combo.show()

        # Device Name parameter
        label = Gtk.Label.new_with_mnemonic(_("_Device Name"))
        grid.attach(label, 2, 0, 1, 1)
        label.show()
        combo = GimpUi.prop_string_combo_box_new(
            config, "device_name", device_name_enum.get_tree_model(), 0, 1
        )
        grid.attach(combo, 3, 0, 1, 1)
        combo.show()

        # Show Logo
        logo = Gtk.Image.new_from_file(image_paths["logo"])
        vbox.pack_start(logo, False, False, 1)
        logo.show()

         # Spinner
        spinner = Gtk.Spinner()
        vbox.pack_start(spinner, False, False, 1)

        progress_bar = Gtk.ProgressBar()
        vbox.add(progress_bar)
        progress_bar.show()

        # Wait for user to click
        dialog.show()

        import threading
        run_inference_thread = None

        while True:
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                scale = config.get_property("scale")
                device_name = config.get_property("device_name")
                model_name = config.get_property("model_name")
                
                runner = SRRunner(procedure, image, layer, scale, device_name, model_name, progress_bar, config_path_output)
                spinner.show()
                spinner.start()
                run_inference_thread = threading.Thread(target=async_run_superes, args=(runner, dialog))
                run_inference_thread.start()
                run_button.set_sensitive(False)
                continue
            elif response == SRDialogResponse.RunInferenceComplete:
                print ("run superres complete")
                spinner.stop()
                spinner.hide()
                if run_inference_thread:
                    run_inference_thread.join()
                    result = runner.result
                    
                    if result == Gimp.PDBStatusType.SUCCESS and config is not None:
                        config.end_run(Gimp.PDBStatusType.SUCCESS)

                    return result
            elif response == Gtk.ResponseType.APPLY:
                url = "https://github.com/intel/openvino-ai-plugins-gimp.git/README.md"
                Gio.app_info_launch_default_for_uri(url, None)
                continue
            else:
                dialog.destroy()
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )

class Superresolution(Gimp.PlugIn):
    ## GimpPlugIn virtual methods ##
    def do_set_i18n(self, procname):
        return True, 'gimp30-python', None
    
    def do_query_procedures(self):
        return ["superresolution-ov"]

    def do_create_procedure(self, name):
        procedure = None
        if name == "superresolution-ov":
            procedure = Gimp.ImageProcedure.new(self, name, 
                                                Gimp.PDBProcType.PLUGIN, 
                                                run, None)
            procedure.set_image_types("*")
            procedure.set_documentation(
                N_("superresolution on the current layer."),
                globals()["__doc__"],
                name,
            )
            procedure.set_menu_label(_("Super Resolution"))
            procedure.set_attribution("Arisha Kumar", "OpenVINO-AI-Plugins", "2022")
            procedure.add_menu_path("<Image>/Layer/OpenVINO-AI-Plugins/")
            procedure.add_int_argument("scale", _("_Scale"), 
                                       "Scale", 1, 4, 2, 
                                       GObject.ParamFlags.READWRITE)
            procedure.add_string_argument("device_name",_("Device Name"),
                                          "Device Name: 'CPU', 'GPU'",
                                          "CPU",
                                          GObject.ParamFlags.READWRITE)
            procedure.add_string_argument("model_name",_("Model Name"),
                                          "Model Name: 'esrgan', 'sr_1033'",
                                          "sr_1033",
                                           GObject.ParamFlags.READWRITE)
        return procedure


Gimp.main(Superresolution.__gtype__, sys.argv)
