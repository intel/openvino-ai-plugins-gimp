#!/usr/bin/env python3
# Copyright(C) 2025 Rupesh Sreeraman
# SPDX - License - Identifier: Apache - 2.0

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp, Gio, GLib

gi.require_version("GimpUi", "3.0")

import json
import os
import sys
import tempfile
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from fastsd_api_client import FastSDApiClient

from gi.repository import GimpUi

FASTSD_SERVER_URL = "http://localhost:8000"


FASTSD_OV_ERROR = GLib.quark_from_string("fastsd-ov-plugin-error")


class FastSDPluginSettings:
    def __init__(
        self,
        inference_steps: int = 1,
        image_width: int = 512,
        image_height: int = 512,
        ov_model_id: str = "",
        prompt: int = "",
        seed: int = None,
        use_seed: bool = False,
        guidance_scale: float = 1.0,
    ):
        self.inference_steps = inference_steps
        self.image_width = image_width
        self.image_height = image_height
        self.ov_model_id = ov_model_id
        self.prompt = prompt
        self.seed = seed
        self.use_seed = use_seed
        self.guidance_scale = guidance_scale

    def to_json(self):
        settings = {
            "prompt": self.prompt,
            "inference_steps": self.inference_steps,
            "use_openvino": True,
            "use_tiny_auto_encoder": False,
            "openvino_lcm_model_id": self.ov_model_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "seed": self.seed,
            "use_seed": self.use_seed,
            "guidance_scale": self.guidance_scale,
        }
        return json.dumps(settings)


class FastSDApiError(Exception):
    """Custom exception for FastSD API communication errors."""

    pass


class ImageProcessingError(Exception):
    """Custom exception for errors during image processing (decode, load)."""

    pass


class FastSDPlugin(Gimp.PlugIn):
    def do_query_procedures(self):
        return ["fastsd-plugin"]

    def init_ui_settings(self):
        with ThreadPoolExecutor() as executor:
            info_future = executor.submit(self.fast_sd_requests.get_info)
            models_future = executor.submit(self.fast_sd_requests.get_models)
            settings_future = executor.submit(self.fast_sd_requests.load_settings)

            self.info = info_future.result()
            self.models = models_future.result()
            self.settings = settings_future.result()

        diffusion_setting = self.settings.get("lcm_diffusion_setting", {})
        self.fastsd_plugin_settings.ov_model_id = diffusion_setting.get(
            "openvino_lcm_model_id", ""
        )
        self.fastsd_plugin_settings.inference_steps = diffusion_setting.get(
            "inference_steps", 1
        )
        self.fastsd_plugin_settings.image_height = diffusion_setting.get(
            "image_height", 512
        )
        self.fastsd_plugin_settings.image_width = diffusion_setting.get(
            "image_width", 512
        )
        self.fastsd_plugin_settings.guidance_scale = diffusion_setting.get(
            "guidance_scale", 1.0
        )
        self.fastsd_plugin_settings.use_seed = diffusion_setting.get("use_seed", False)
        if self.fastsd_plugin_settings.use_seed:
            self.fastsd_plugin_settings.seed = diffusion_setting.get("seed", 42)

    def update_ui(self):
        self.set_sensitive(True)
        self.inference_scale.set_value(self.fastsd_plugin_settings.inference_steps)
        self.width_combo.set_active(
            self.find_index_by_text(
                self.width_combo, str(self.fastsd_plugin_settings.image_width)
            )
        )
        self.height_combo.set_active(
            self.find_index_by_text(
                self.height_combo, str(self.fastsd_plugin_settings.image_height)
            )
        )
        self.guidance_scale.set_value(self.fastsd_plugin_settings.guidance_scale)

        if self.models:
            for model in self.models:
                self.model_combo.append_text(model)

            if self.fastsd_plugin_settings.ov_model_id in self.models:
                index = self.models.index(self.fastsd_plugin_settings.ov_model_id)
                self.model_combo.set_active(index)
            else:
                self.model_combo.set_active(0)
        self.device_label.set_text(
            f"{self.info.get('device_type', '').upper()} : {self.info.get('device_name', '')}"
        )
        if self.fastsd_plugin_settings.use_seed:
            self.seed.set_text(str(self.fastsd_plugin_settings.seed))
        else:
            self.seed.set_text("")

    def generate_image(self, config):
        image_generation_future = self.executor.submit(
            self.fast_sd_requests.generate_text_to_image,
            config,
        )
        image_generation_future.add_done_callback(self._on_image_generated)

    def _on_image_generated(self, image_generation_future):
        try:
            result = image_generation_future.result()
            base64_image = result["images"][0]
            image_data = b64decode(base64_image)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_file.write(image_data)
            temp_file.close()

            temp_file_path = temp_file.name

            GLib.idle_add(
                self._handle_image_generated,
                temp_file_path,
            )

        except Exception as e:
            print(f"Error in background task: {e}")

    def _handle_image_generated(self, image_path):
        try:
            self.load_image_to_gimp(image_path)
        except Exception as e:
            Gimp.message(f"Error loading image: {e}")

        self.generate_button.set_sensitive(True)
        self.generate_button.set_label("Generate")
        return False

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run, None
        )
        procedure.set_image_types("*")
        procedure.set_menu_label("FastSD")
        procedure.add_menu_path("<Image>/Layer/OpenVINO-AI-Plugins/")
        procedure.set_documentation(
            "Stablediffusion on the current layer using FastSD - OpenVINO.",
            "Stablediffusion on the current layer using FastSD - OpenVINO.",
            name,
        )
        procedure.set_attribution("Rupesh Sreeraman", "FastSD", "2025")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.ALWAYS)
        return procedure

    def find_index_by_text(self, combo, target_text):
        index = 0
        model = combo.get_model()
        for row in model:
            if row[0] == target_text:
                return index
            index += 1
        return -1

    def set_sensitive(self, sensitive: bool):
        self.textview.set_sensitive(sensitive)
        self.model_combo.set_sensitive(sensitive)
        self.inference_scale.set_sensitive(sensitive)
        self.width_combo.set_sensitive(sensitive)
        self.height_combo.set_sensitive(sensitive)
        self.generate_button.set_sensitive(sensitive)

    def get_gen_settings(self):
        prompt = self.textview.get_buffer().get_text(
            self.textview.get_buffer().get_start_iter(),
            self.textview.get_buffer().get_end_iter(),
            False,
        )

        inference_steps = int(self.inference_scale.get_value())
        selected_model = self.model_combo.get_active_text()
        self.fastsd_plugin_settings.ov_model_id = selected_model
        self.fastsd_plugin_settings.inference_steps = inference_steps
        self.fastsd_plugin_settings.prompt = prompt
        self.fastsd_plugin_settings.image_width = int(
            self.width_combo.get_active_text()
        )
        self.fastsd_plugin_settings.image_height = int(
            self.height_combo.get_active_text()
        )
        self.fastsd_plugin_settings.guidance_scale = round(
            self.guidance_scale.get_value(), 1
        )

        return self.fastsd_plugin_settings.to_json()

    def load_image_to_gimp(
        self,
        image_path: str,
    ):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        try:
            image_new = Gimp.Image.new(
                self.fastsd_plugin_settings.image_width,
                self.fastsd_plugin_settings.image_height,
                0,
            )
            display = Gimp.Display.new(image_new)
            result = Gimp.file_load(
                Gimp.RunMode.NONINTERACTIVE,
                Gio.file_new_for_path(image_path),
            )
            result_layer = result.get_layers()[0]
            copy = Gimp.Layer.new_from_drawable(result_layer, image_new)
            set_name = "fastsd-plugin-image"
            copy.set_name(set_name)
            copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)
            image_new.insert_layer(copy, None, -1)
            Gimp.displays_flush()
        except Exception as e:
            raise ImageProcessingError(f"Error loading image {e}")

    def _is_valid_seed(self, value: str):
        try:
            seed = int(value)
            return 0 <= seed <= 999999999
        except (ValueError, TypeError):
            return False

    def run(
        self,
        procedure,
        run_mode,
        image,
        drawables,
        config,
        run_data,
    ):
        self.info = {}
        self.models = {}
        self.settings = {}
        self.fast_sd_requests = FastSDApiClient(FASTSD_SERVER_URL)
        self.fastsd_plugin_settings = FastSDPluginSettings()
        self.file_path = None
        self.executor = ThreadPoolExecutor()
        logo_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "openvino_utils",
            "images",
            "plugin_logo.png",
        )

        if run_mode == Gimp.RunMode.INTERACTIVE:
            gi.require_version("Gtk", "3.0")
            from gi.repository import Gtk

            def show_error_dialog(parent, message: str):
                dialog = Gtk.MessageDialog(
                    parent=parent,
                    flags=0,
                    message_type=Gtk.MessageType.ERROR,
                    buttons=Gtk.ButtonsType.OK,
                    text="FastSD - OpenVINO",
                    secondary_text="Error",
                )
                dialog.format_secondary_text(message)
                dialog.run()
                dialog.destroy()

            GimpUi.init("fastsd-plugin")

            dialog = GimpUi.Dialog(
                title="FastSD - OpenVINO",
                role="fastsd-plugin",
                use_header_bar=False,
            )
            dialog.set_default_size(400, 400)
            dialog.set_resizable(False)

            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
            vbox.set_margin_top(10)
            vbox.set_margin_bottom(10)
            vbox.set_margin_start(10)
            vbox.set_margin_end(10)
            logo = Gtk.Image.new_from_file(logo_path)
            vbox.pack_start(logo, False, False, 0)

            self.device_label = Gtk.Label(
                label=f"{self.info.get('device_type', '').upper()} : {self.info.get('device_name', '')}"
            )
            self.device_label.set_text("Loading settings please wait...")
            vbox.pack_start(self.device_label, False, False, 0)

            prompt_label = Gtk.Label(label="Describe the image you want to generate :")
            prompt_label.set_halign(Gtk.Align.START)
            prompt_label.set_valign(Gtk.Align.CENTER)
            vbox.pack_start(prompt_label, False, False, 0)
            self.textview = Gtk.TextView()
            self.textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            self.textview.set_size_request(-1, 50)
            vbox.pack_start(self.textview, False, False, 0)

            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            vbox.pack_start(hbox, False, False, 0)

            self.generate_button = Gtk.Button(label="Generate")
            self.generate_button.set_size_request(100, -1)
            hbox.pack_start(Gtk.Label(), True, True, 0)
            hbox.pack_start(self.generate_button, False, False, 0)

            # Create a grid
            grid = Gtk.Grid()
            grid.set_row_spacing(5)
            grid.set_column_spacing(10)

            model_label = Gtk.Label(label="Model:")
            model_label.set_halign(Gtk.Align.START)
            model_label.set_valign(Gtk.Align.CENTER)
            self.model_combo = Gtk.ComboBoxText()

            grid.attach(model_label, 0, 0, 1, 1)
            grid.attach(self.model_combo, 1, 0, 1, 1)

            inference_label = Gtk.Label(label="Number of inference steps:")
            inference_label.set_halign(Gtk.Align.START)
            inference_label.set_valign(Gtk.Align.CENTER)

            self.inference_scale = Gtk.SpinButton()
            self.inference_scale.set_range(0, 50)
            self.inference_scale.set_value(self.fastsd_plugin_settings.inference_steps)
            self.inference_scale.set_increments(1, 10)
            self.inference_scale.set_input_purpose(Gtk.InputPurpose.NUMBER)

            grid.attach(inference_label, 0, 1, 1, 1)
            grid.attach(self.inference_scale, 1, 1, 1, 1)

            width_label = Gtk.Label(label="Image Width:")
            width_label.set_halign(Gtk.Align.START)
            width_label.set_valign(Gtk.Align.CENTER)

            self.width_combo = Gtk.ComboBoxText()
            self.width_combo.append_text("256")
            self.width_combo.append_text("512")
            self.width_combo.append_text("768")
            self.width_combo.append_text("1024")
            self.width_combo.set_active(0)

            grid.attach(width_label, 0, 2, 1, 1)
            grid.attach(self.width_combo, 1, 2, 1, 1)

            height_label = Gtk.Label(label="Image Height:")
            height_label.set_halign(Gtk.Align.START)
            height_label.set_valign(Gtk.Align.CENTER)

            self.height_combo = Gtk.ComboBoxText()
            self.height_combo.append_text("256")
            self.height_combo.append_text("512")
            self.height_combo.append_text("768")
            self.height_combo.append_text("1024")
            self.height_combo.set_active(0)

            grid.attach(height_label, 0, 3, 1, 1)
            grid.attach(self.height_combo, 1, 3, 1, 1)

            seed_label = Gtk.Label(label="Seed :")
            seed_label.set_halign(Gtk.Align.START)
            seed_label.set_valign(Gtk.Align.CENTER)

            self.seed = Gtk.Entry.new()
            self.seed.set_placeholder_text("If left blank, random seed will be set..")

            grid.attach(seed_label, 0, 4, 1, 1)
            grid.attach(self.seed, 1, 4, 1, 1)

            guidance_label = Gtk.Label(label="Guidance Scale:")
            guidance_label.set_halign(Gtk.Align.START)
            guidance_label.set_valign(Gtk.Align.CENTER)

            self.guidance_scale = Gtk.SpinButton()
            self.guidance_scale.set_range(0, 10.0)
            self.guidance_scale.set_value(self.fastsd_plugin_settings.guidance_scale)
            self.guidance_scale.set_increments(0.1, 0.1)
            self.guidance_scale.set_digits(1)
            self.guidance_scale.set_input_purpose(Gtk.InputPurpose.NUMBER)

            grid.attach(guidance_label, 0, 5, 1, 1)
            grid.attach(self.guidance_scale, 1, 5, 1, 1)

            vbox.pack_start(grid, False, False, 0)

            separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
            vbox.pack_start(separator, expand=False, fill=True, padding=5)

            dialog.add_button("_Help", Gtk.ResponseType.HELP)
            dialog.add_button("_OK", Gtk.ResponseType.OK)

            self.set_sensitive(False)

            try:
                self.init_ui_settings()
                self.update_ui()
            except RuntimeError as exp:
                return procedure.new_return_values(
                    Gimp.PDBStatusType.EXECUTION_ERROR,
                    GLib.Error.new_literal(
                        domain=FASTSD_OV_ERROR,
                        code=1,
                        message=f"Ensure that FastSD server is running at {FASTSD_SERVER_URL}.\n{str(exp)}",
                    ),
                )

            def on_generate_clicked(button):
                if self.seed.get_text() != "":
                    seed_val = self.seed.get_text()
                    if self._is_valid_seed(seed_val):
                        self.fastsd_plugin_settings.seed = int(seed_val)
                        self.fastsd_plugin_settings.use_seed = True
                    else:
                        show_error_dialog(
                            dialog,
                            "Error : Invalid seed value, please enter a number or leave seed blank for random seed",
                        )
                        return
                else:
                    self.fastsd_plugin_settings.use_seed = False

                self.generate_button.set_sensitive(False)
                self.generate_button.set_label("Generating...")
                self.generate_image(self.get_gen_settings())

            self.generate_button.connect("clicked", on_generate_clicked)

            dialog.get_content_area().add(vbox)
            dialog.show_all()
            _ = dialog.run()
            dialog.destroy()

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


Gimp.main(FastSDPlugin.__gtype__, sys.argv)
