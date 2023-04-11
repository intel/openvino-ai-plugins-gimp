# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

#!/usr/bin/env python3
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
#import pickle
import os
import sys


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

model_name_enum = StringEnum(
    "SD_1.4",
    _("SD_1.4"),
    "SD_1.5",
    _("SD_1.5"),
)  

scheduler_name_enum = StringEnum(
    "LMSDiscreteScheduler",
    _("LMSDiscreteScheduler"),
    "PNDMScheduler",
    _("PNDMScheduler"),
    "EulerDiscreteScheduler",
    _("EulerDiscreteScheduler"),
)   
    

def stablediffusion(procedure, image, drawable, device_name, prompt, negative_prompt, num_infer_steps, guidance_scale, initial_image, strength, seed, create_gif,scheduler,model_name, progress_bar, config_path_output):
    # Save inference parameters and layers
    weight_path = config_path_output["weight_path"]
    python_path = config_path_output["python_path"]
    plugin_path = config_path_output["plugin_path"]

    Gimp.context_push()
    #image.undo_group_start()
    print("FIRST IMAGE WHAT IS IT:",type(image))
    print("FIRST drawable WHAT IS IT:",type(drawable))
    #save_image(image, drawable, os.path.join(weight_path, "..", "cache.png"))
    with open(os.path.join(weight_path, "..", "gimp_openvino_run.json"), "w") as file:
        json.dump({"device_name": device_name,"prompt": prompt, "negative_prompt": negative_prompt, "num_infer_steps": num_infer_steps,"guidance_scale": guidance_scale,"initial_image": initial_image, "strength": strength, "seed": seed, "create_gif": create_gif, "scheduler": scheduler, "model_name": model_name, "inference_status": "started"}, file)


    # Run inference and load as layer
    subprocess.call([python_path, plugin_path])

    with open(os.path.join(weight_path, "..", "gimp_openvino_run.json"), "r") as file:
        data_output = json.load(file)
    #image.undo_group_end()
    Gimp.context_pop()
    #scale = 3
    if data_output["inference_status"] == "success":
        #if initial_image == None:
        #    image_new = Gimp.Image.new(
        #            drawable[0].get_width(), drawable[0].get_height() , 0
        #        )  
        #else:
        image_new = Gimp.Image.new(
                    data_output["src_width"], data_output["src_height"] , 0
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
        #copy = Gimp.Layer.new_from_drawable(result_layer, image)
        copy = Gimp.Layer.new_from_drawable(result_layer, image_new)
        copy.set_name("Stable Diffusion")
        copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)  # DIFFERENCE_LEGACY
        image_new.insert_layer(copy, None, -1)

        Gimp.displays_flush()
        #image.undo_group_end()
        Gimp.context_pop()


        # Remove temporary layers that were saved
        my_dir = os.path.join(weight_path, "..")
        #for f_name in os.listdir(my_dir):
        #    if f_name.startswith("cache"):
        #        os.remove(os.path.join(my_dir, f_name))

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    else:
        show_dialog(
            "Inference not successful. See error_log.txt in GIMP-OpenVINO folder.",
            "Error !",
            "error",
            image_paths
        )
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


def run(procedure, run_mode, image, n_drawables, layer, args, data ):
    model_name = args.index(0)
    device_name = args.index(1)
    prompt = args.index(2)
    scheduler = args.index(3)
    negative_prompt = args.index(4)
    num_infer_steps = args.index(5)
    guidance_scale = args.index(6)
    seed = args.index(7)
    create_gif = args.index(8)
    initial_image = args.index(9)
    strength = args.index(10)

    if run_mode == Gimp.RunMode.INTERACTIVE:
        # Get all paths
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "tools"
        )
        
        
        with open(os.path.join(config_path, "gimp_openvino_config.json"), "r") as file:
            config_path_output = json.load(file)
        python_path = config_path_output["python_path"]
        config_path_output["plugin_path"] = os.path.join(config_path, "stable-diffusion-ov.py")
        
        device_name_enum = DeviceEnum(config_path_output["supported_devices"])

        config = procedure.create_config()
        config.begin_run(image, run_mode, args)
        

        GimpUi.init("stable-diffusion-ov.py")
        use_header_bar = Gtk.Settings.get_default().get_property(
            "gtk-dialogs-use-header"
        )
        dialog = GimpUi.Dialog(use_header_bar=use_header_bar, title=_("Stable Diffusion...."))
        dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("_Help", Gtk.ResponseType.APPLY)
        dialog.add_button("_Run Inference", Gtk.ResponseType.OK)

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

        #Prompt text
        prompt_text = Gtk.Entry.new()
        grid.attach(prompt_text, 1, 1, 1, 1)
        prompt_text.set_width_chars(60)
      
        prompt_text.show()

        prompt_text_label = _("Enter text to generate image")
        prompt_label = Gtk.Label(label=prompt_text_label)
        grid.attach(prompt_label, 0, 1, 1, 1)
        vbox.pack_start(prompt_label, False, False, 1)
        prompt_label.show()

        # Scheduler Name parameter
        label = Gtk.Label.new_with_mnemonic(_("_Scheduler"))
        grid.attach(label, 2, 1, 1, 1)
        label.show()
        combo = GimpUi.prop_string_combo_box_new(
            config, "scheduler", scheduler_name_enum.get_tree_model(), 0, 1
        )
        grid.attach(combo, 3, 1, 1, 1)
        combo.show()   
        
        #Negative Prompt text
        negative_prompt_text = Gtk.Entry.new()
        grid.attach(negative_prompt_text, 1, 2, 1, 1)
        negative_prompt_text.set_width_chars(60)
      
        negative_prompt_text.show()

        negative_prompt_text_label = _("Negative Prompt")
        negative_prompt_label = Gtk.Label(label=negative_prompt_text_label)
        grid.attach(negative_prompt_label, 0, 2, 1, 1)
        vbox.pack_start(negative_prompt_label, False, False, 1)
        negative_prompt_label.show()        

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
        #seed.set_text("Person|Cars")
        seed.show()

        seed_text = _("Seed")
        seed_label = Gtk.Label(label=seed_text)
        grid.attach(seed_label, 0, 5, 1, 1)
        vbox.pack_start(seed_label, False, False, 1)
        seed_label.show()

        # Create GIF Parameter
        #config.set_property("create_gif", create_gif)
        spin = GimpUi.prop_check_button_new(config, "create_gif", _("_Create GIF from the latent frames generated at each inference step"))
        spin.set_tooltip_text(
            _(
                "If checked, a GIF of all the latent frames will be created. "
                "Please note that currently this process is extremely slow."
             
            )
        )
        grid.attach(spin, 1, 6, 1, 1)
        spin.show()
        
        #UI to browse Initial Image 
        def choose_file(widget):
            if file_chooser_dialog.run() == Gtk.ResponseType.OK:
                if file_chooser_dialog.get_file() is not None:
                    #config.set_property("file", file_chooser_dialog.get_file())
                    file_entry.set_text(file_chooser_dialog.get_file().get_path())
                
            file_chooser_dialog.hide()

        file_chooser_button = Gtk.Button.new_with_mnemonic(label=_("_Init Image (Optional)..."))
        grid.attach(file_chooser_button, 0, 7, 1, 1)
        file_chooser_button.show()
        file_chooser_button.connect("clicked", choose_file)

        file_entry = Gtk.Entry.new()
        grid.attach(file_entry, 1, 7, 1, 1)
        file_entry.set_width_chars(40)
        file_entry.set_placeholder_text(_("Choose path..."))
        if initial_image is not None:
            #print("initial_image",initial_image)
            file_entry.set_text(initial_image.get_path())
        file_entry.show()

        file_chooser_dialog = Gtk.FileChooserDialog(
            use_header_bar=use_header_bar,
            title=_("Initial Image path..."),
            action=Gtk.FileChooserAction.OPEN,
        )
        file_chooser_dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        file_chooser_dialog.add_button("_OK", Gtk.ResponseType.OK)

        # Initial_image strength parameter
        label = Gtk.Label.new_with_mnemonic(_("_Strength of Initial Image"))
        grid.attach(label, 0, 8, 1, 1)
        label.show()
        spin = GimpUi.prop_spin_button_new(
            config, "strength", step_increment=0.1, page_increment=0.1, digits=1
        )
        grid.attach(spin, 1, 8, 1, 1)
        spin.show()

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

        # Wait for user to click
        dialog.show()
        while True:
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                device_name = config.get_property("device_name")
                prompt = prompt_text.get_text()
                negative_prompt = negative_prompt_text.get_text()
                num_infer_steps = config.get_property("num_infer_steps")
                guidance_scale = config.get_property("guidance_scale")
                strength = config.get_property("strength")
                model_name = config.get_property("model_name")
                scheduler = config.get_property("scheduler")


                if len(file_entry.get_text()) != 0:
                    initial_image = file_entry.get_text()
                else:
                    initial_image = None
                if len(seed.get_text()) != 0:
                    seed = seed.get_text()
                else:
                    seed = None
                create_gif = config.get_property("create_gif")
                
                result = stablediffusion(
                    procedure, image, layer, device_name, prompt, negative_prompt, num_infer_steps, guidance_scale, initial_image, strength, seed, create_gif,scheduler,model_name, progress_bar, config_path_output
                )
                
                # super_resolution(procedure, image, n_drawables, layer, force_cpu, progress_bar, config_path_output)
                # If the execution was successful, save parameters so they will be restored next time we show dialog.
                #if result.index(0) == Gimp.PDBStatusType.SUCCESS and config is not None:
                #    config.end_run(Gimp.PDBStatusType.SUCCESS)
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


class StableDiffusion(Gimp.PlugIn):
    ## Parameters ##
    __gproperties__ = {
        "num_infer_steps": (int, _("_Number of Inference steps (Default:32)"), "Number of Inference steps (Default:32)", 1, 50, 32, GObject.ParamFlags.READWRITE,),
        "guidance_scale": (float, _("_Guidance Scale (Default:7.5)"), "_Guidance Scale (Default:7.5)", 1.0, 20.0, 7.5, GObject.ParamFlags.READWRITE,),
        "strength": (float, _("_Strength of Initial Image (Default:0.8)"), "_Strength of Initial Image (Default:0.8)", 0.0, 1.0, 0.8, GObject.ParamFlags.READWRITE,),         
        "create_gif": (
            bool, 
            _("_Create GIF from the latent frames generated at each inference step"), 
            "Create GIF", 
            False, 
            GObject.ParamFlags.READWRITE,
            ),
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
            procedure.add_argument_from_property(self, "device_name")
            #procedure.add_argument_from_property(self, "initial_image")
            procedure.add_argument_from_property(self, "num_infer_steps")
            procedure.add_argument_from_property(self, "guidance_scale")
            procedure.add_argument_from_property(self, "strength")
            procedure.add_argument_from_property(self, "create_gif")
            procedure.add_argument_from_property(self, "model_name")
            procedure.add_argument_from_property(self, "scheduler")
            
        return procedure


Gimp.main(StableDiffusion.__gtype__, sys.argv)
