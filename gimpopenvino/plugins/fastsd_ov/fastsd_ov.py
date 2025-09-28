#!/usr/bin/env python3
# Copyright(C) 2022-2025 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0


import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp, Gio, GLib, Pango, Gtk

import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread


sys.path.extend(
    [os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "openvino_utils")]
)
from gi.repository import GimpUi
from tools.openvino_common.models_ov.fastsd.model_config import ModelConfig
from tools.tools_utils import SDOptionCache, config_path_dir

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server
MODEL_DISPLAY_TEXT_MAX_LENGTH = 40
STABLE_DIFFUSION_OV_SERVER = "stable_diffusion_ov_server.py"
CONFIG_FILE = os.path.join(config_path_dir, "fastsd_models.json")
FASTSD_CACHE_CONFIG = "gimp_openvino_run_fastsd.json"


class ModelManagerDialog(Gtk.Dialog):
    """Dialog to add/remove models (persistent)."""

    def __init__(self, parent=None):
        super().__init__(title="FastSD Model Manager", transient_for=parent, flags=0)

        self.set_default_size(400, 300)
        self.set_resizable(False)
        try:
            self.models_config = ModelConfig(CONFIG_FILE)
            config = self.models_config.load()
            self.models = config.get("models", [])
        except Exception as exc:
            self.models = []

        box = self.get_content_area()
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox.set_margin_top(5)
        vbox.set_margin_bottom(5)
        vbox.set_margin_start(5)
        vbox.set_margin_end(5)
        box.add(vbox)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_min_content_height(200)
        vbox.pack_start(scroll, True, True, 0)

        self.listbox = Gtk.ListBox()
        scroll.add(self.listbox)
        self.refresh_list()

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.entry = Gtk.Entry(
            placeholder_text="Enter HuggingFace model path or local model path"
        )
        hbox.pack_start(self.entry, True, True, 0)

        add_btn = Gtk.Button(label="Add")
        add_btn.connect("clicked", self.on_add_clicked)
        hbox.pack_start(add_btn, False, False, 0)
        vbox.pack_start(hbox, False, False, 0)

        reset_button = Gtk.Button(label="_Reset", use_underline=True)
        reset_button.connect("clicked", self.on_reset_clicked)
        self.get_action_area().pack_start(reset_button, False, False, 0)
        self.add_button("_Close", Gtk.ResponseType.CLOSE)
        self.connect("response", self.on_response)
        self.show_all()

    def on_reset_clicked(self, button):
        self.models = self.models_config.get_default_models()
        self.models_config.save("models", self.models)
        self.refresh_list()

    def on_response(self, dialog, response_id):
        if response_id in (Gtk.ResponseType.CLOSE, Gtk.ResponseType.DELETE_EVENT):
            self.destroy()

    def refresh_list(self):
        for child in self.listbox.get_children():
            self.listbox.remove(child)

        for model in self.models:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            label = Gtk.Label(label=model, xalign=0)
            label.set_ellipsize(Pango.EllipsizeMode.END)
            if len(model) > MODEL_DISPLAY_TEXT_MAX_LENGTH:
                label.set_max_width_chars(MODEL_DISPLAY_TEXT_MAX_LENGTH)
                label.set_tooltip_text(model)
            row.pack_start(label, True, True, 0)

            del_btn = Gtk.Button(label="Delete")
            del_btn.connect("clicked", self.on_delete_clicked, model)
            row.pack_start(del_btn, False, False, 0)

            row.show_all()
            self.listbox.add(row)

    def on_add_clicked(self, button):
        new_model = self.entry.get_text().strip()
        if new_model and new_model not in self.models:
            self.models.append(new_model)
            self.models_config.save("models", self.models)
            self.entry.set_text("")
            self.refresh_list()
            self.show_all()

    def on_delete_clicked(self, button, model):
        if model in self.models:
            self.models.remove(model)
            self.models_config.save("models", self.models)
            self.refresh_list()
            self.show_all()


def is_server_running():
    HOST = "127.0.0.1"
    PORT = 65432

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(
                0.1
            )  # <- set connection timeout to 100 ms (default is a few seconds)
            sock.connect((HOST, PORT))
            sock.sendall(b"ping")
            data = sock.recv(1024)
            if data.decode() == "ping":
                return True
    except Exception as exc:
        return False


def get_bin_dir(python_exe: str) -> Path:
    """
    Given the full path to a Python executable inside a venv,
    return the path to its bin/Scripts directory.

    Works for Windows (Scripts) and Linux/macOS (bin).
    """
    exe_path = Path(python_exe).resolve()
    return exe_path.parent


class SDRunner:
    def __init__(self, procedure, image, drawable, config_path_output):
        self.procedure = procedure
        self.image = image
        self.drawable = drawable
        self.config_path_output = config_path_output
        self.result = None
        self.sd_data = SDOptionCache(
            os.path.join(config_path_output["weight_path"], "..", FASTSD_CACHE_CONFIG)
        )
        self.saved_seed = self.sd_data.get("seed")
        self.seed = self.saved_seed

    def run(self, dialog):
        procedure = self.procedure
        image = self.image
        config_path_output = self.config_path_output
        weight_path = config_path_output["weight_path"]
        Gimp.context_push()

        if image:
            image.undo_group_start()

        if self.seed is None:
            self.sd_data.set("seed", None)
            self.sd_data.save()

        self.current_step = 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(b"go")
            while True:
                data = s.recv(1024)
                response = data.decode()
                if response == "done":
                    break

        self.sd_data.load()

        if image:
            image.undo_group_end()
        Gimp.context_pop()

        if self.sd_data.get("inference_status") == "success":
            image_new = Gimp.Image.new(
                self.sd_data.get("src_width"), self.sd_data.get("src_height"), 0
            )
            display = Gimp.Display.new(image_new)
            cache_image = "sd_cache.png"
            result = Gimp.file_load(
                Gimp.RunMode.NONINTERACTIVE,
                Gio.file_new_for_path(os.path.join(weight_path, "..", cache_image)),
            )
            result_layer = result.get_layers()[0]

            copy = Gimp.Layer.new_from_drawable(result_layer, image_new)
            set_name = "Stable Diffusion -" + str(self.sd_data.get("seed"))
            copy.set_name(set_name)
            copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)  # DIFFERENCE_LEGACY
            image_new.insert_layer(copy, None, -1)

            Gimp.displays_flush()
            if image:
                image.undo_group_end()
            Gimp.context_pop()

            # Remove temporary layers that were saved
            my_dir = os.path.join(weight_path, "..")
            for f_name in os.listdir(my_dir):
                if f_name.startswith("sd_cache"):
                    os.remove(os.path.join(my_dir, f_name))

            self.result = procedure.new_return_values(
                Gimp.PDBStatusType.SUCCESS, GLib.Error()
            )
            self.sd_data.set("seed", self.saved_seed)
            self.sd_data.save()
            dialog.response(Gtk.ResponseType.OK)
            return self.result

        else:
            dialog.response(Gtk.ResponseType.OK)
            msg_dlg = Gtk.MessageDialog(
                parent=dialog,
                flags=0,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="FastSD - OpenVINO",
                secondary_text="Info",
            )
            msg_dlg.format_secondary_text(
                "Inference not successful. See error_log.txt in GIMP-OpenVINO folder."
            )
            msg_dlg.run()
            msg_dlg.destroy()
            self.show_error_dialog(
                dialog,
            )

            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


class FastSDPlugin(Gimp.PlugIn):
    def do_query_procedures(self):
        return ["fastsd-plugin"]

    def _load_model_config(self):
        self.config = self.models_config.load()
        self.models = self.config.get("models", [])
        self.device_name = self.config.get("device_name", "CPU")

    def _get_current_device(self):
        return self.config.get("device_name", "CPU")

    def init_ui_settings(self):
        self.models_config = ModelConfig(CONFIG_FILE)
        self._load_model_config()
        for device in self.config_path_output["supported_devices"]:
            if "GNA" not in device:
                self.supported_devices.append(device)

    def update_ui(self):
        self.set_sensitive(True)
        if os.path.exists(self._get_sd_options_path()):
            options = SDOptionCache(self._get_sd_options_path())
            prompt = options.get("prompt")
            guidance_scale = options.get("guidance_scale")
            num_infer_steps = options.get("num_infer_steps")
            src_height = options.get("src_height")
            src_width = options.get("src_width")
            self.model_path = options.get("model_name")
            seed = options.get("seed")
        else:
            prompt = ""
            guidance_scale = 1.0
            num_infer_steps = 1
            src_height = 512
            src_width = 512
            self.model_path = ""
            seed = None

        buffer = self.prompt_text.get_buffer()
        buffer.set_text(prompt)
        self.inference_scale.set_value(num_infer_steps)
        self.width_combo.set_active(
            self.find_index_by_text(self.width_combo, str(src_width))
        )
        self.height_combo.set_active(
            self.find_index_by_text(self.height_combo, str(src_height))
        )
        self.guidance_scale.set_value(guidance_scale)
        store = Gtk.ListStore(str, str)
        if self.models:
            for model in self.models:
                if len(model) > MODEL_DISPLAY_TEXT_MAX_LENGTH:
                    short_model = model[:MODEL_DISPLAY_TEXT_MAX_LENGTH] + "..."
                else:
                    short_model = model
                store.append([short_model, model])

            self.model_combo.set_model(store)
            self.model_combo.set_active(0)

        index = self.find_index_by_text(self.model_combo, self.model_path, 1)
        if index != -1:
            self.model_combo.set_active(index)
        else:
            self.model_combo.set_active(0)

        if seed:
            self.seed.set_text(str(seed))
        else:
            self.seed.set_text("")

        for device_name in self.supported_devices:
            self.device_combo.append_text(device_name)

        self._update_devices_combo(self.get_model_full_path())

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

    def find_index_by_text(self, combo, target_text, row_index=0):
        index = 0
        model = combo.get_model()
        for row in model:
            if row[row_index] == target_text:
                return index
            index += 1
        return -1

    def async_load_models(
        self,
        python_path,
        server_path,
        model_name,
        supported_devices,
        device_power_mode,
        show_console,
        dialog,
    ):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, PORT))
            s.sendall(b"kill")

            print("stable-diffusion model server killed")
        except:
            print("No stable-diffusion model server found to kill")

        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW
        else:
            creationflags = 0  # N/A on linux

        process = subprocess.Popen(
            [
                python_path,
                server_path,
                model_name,
                str(supported_devices),
                device_power_mode,
            ],
            cwd=str(get_bin_dir(python_path)),
            close_fds=True,
        )

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
                            GLib.idle_add(lambda: self.on_model_ready())
                            break
                    break

    def on_model_ready(self):
        self.is_model_loaded = True
        self.set_sensitive(True)

    def set_sensitive(
        self,
        sensitive: bool,
    ):
        self.generate_button.set_sensitive(sensitive)
        self.prompt_text.set_sensitive(sensitive)
        self.model_combo.set_sensitive(sensitive)
        self.inference_scale.set_sensitive(sensitive)
        self.width_combo.set_sensitive(sensitive)
        self.height_combo.set_sensitive(sensitive)
        self.seed.set_sensitive(sensitive)
        self.guidance_scale.set_sensitive(sensitive)
        self.edit_models_btn.set_sensitive(sensitive)
        self.load_model_button.set_sensitive(sensitive)
        self.device_combo.set_sensitive(sensitive)

    def update_gen_settings(self, sd_option_cache):
        prompt = self.prompt_text.get_buffer().get_text(
            self.prompt_text.get_buffer().get_start_iter(),
            self.prompt_text.get_buffer().get_end_iter(),
            False,
        )
        sd_option_cache.set("model_name", self.model_path)
        sd_option_cache.set("prompt", prompt)
        sd_option_cache.set("guidance_scale", round(self.guidance_scale.get_value(), 1))
        inference_steps = int(self.inference_scale.get_value())
        sd_option_cache.set("num_infer_steps", inference_steps)
        sd_option_cache.set("src_width", int(self.width_combo.get_active_text()))
        sd_option_cache.set("src_height", int(self.height_combo.get_active_text()))
        self.models_config.save("device_name", self.device_combo.get_active_text())
        seed = self.seed.get_text()
        if self._is_valid_seed(seed):
            sd_option_cache.set("seed", str(seed))
        else:
            sd_option_cache.set("seed", None)

        sd_option_cache.save()

    def _is_valid_seed(self, value: str) -> bool:
        try:
            seed = int(value)
            return 0 <= seed <= 999999999
        except (ValueError, TypeError):
            return False

    def _async_sd_run_func(
        self,
        runner,
        dialog,
    ):
        print("Running SD async")
        runner.run(dialog)
        print("async SD done")

    def _get_plugin_version(self):
        try:
            with open(
                os.path.join(config_path_dir, "gimp_openvino_config.json"),
                "r",
            ) as file:
                config_path_output = json.load(file)

            return config_path_output.get("plugin_version")
        except Exception:
            return None

    def _get_sd_options_path(self):
        config_path = os.path.join(
            self.config_path_output["weight_path"], "..", FASTSD_CACHE_CONFIG
        )
        return config_path

    def _update_devices_combo(
        self,
        model_name,
    ):
        cur_device = self.device_combo.get_active_text()
        devices = self.supported_devices.copy()
        if "square" not in model_name.lower():
            self.device_combo.remove_all()
            if "NPU" in devices:
                devices.remove("NPU")
            if "GPU" in devices:
                if "int8" in model_name.lower():
                    devices.remove("GPU")
                if model_name == "rupeshs/sana-sprint-0.6b-openvino-int4":
                    devices.remove("GPU")
            for device_name in devices:
                self.device_combo.append_text(device_name)
        cur_index = self.find_index_by_text(self.device_combo, cur_device)
        self.device_combo.set_active(cur_index if cur_index != -1 else 0)

    def get_model_full_path(self):
        tree_iter = self.model_combo.get_active_iter()
        if tree_iter is not None:
            model = self.model_combo.get_model()
            model_path = model[tree_iter][1]
            return model_path
        return None

    def run(
        self,
        procedure,
        run_mode,
        image,
        layer,
        config,
        run_data,
    ):
        with open(
            os.path.join(config_path_dir, "gimp_openvino_config.json"), "r"
        ) as file:
            self.config_path_output = json.load(file)

        python_path = self.config_path_output["python_path"]
        plugin_version = self.config_path_output["plugin_version"]
        sd_option_cache = SDOptionCache(self._get_sd_options_path())
        self.supported_devices = []
        self.model_path = None
        self.is_model_changed = False
        self.executor = ThreadPoolExecutor()
        self.init_ui_settings()

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
            plugin_version = self._get_plugin_version()
            if plugin_version:
                plugin_title = f"FastSD - OpenVINO : {plugin_version}"
            else:
                plugin_title = "FastSD - OpenVINO"

            dialog = GimpUi.Dialog(
                title=plugin_title,
                role="fastsd-plugin",
                use_header_bar=False,
            )
            dialog.set_default_size(450, 400)
            dialog.set_resizable(False)

            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
            vbox.set_margin_top(20)
            vbox.set_margin_bottom(20)
            vbox.set_margin_start(20)
            vbox.set_margin_end(20)
            logo = Gtk.Image.new_from_file(logo_path)
            vbox.pack_start(logo, False, False, 0)

            prompt_label = Gtk.Label(label="Describe the image you want to generate")
            prompt_label.set_halign(Gtk.Align.START)
            prompt_label.set_valign(Gtk.Align.CENTER)
            vbox.pack_start(prompt_label, False, False, 0)
            self.prompt_text = Gtk.TextView()
            self.prompt_text.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            self.prompt_text.set_size_request(-1, 70)
            vbox.pack_start(self.prompt_text, False, False, 0)

            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            vbox.pack_start(hbox, False, False, 0)

            device_label = Gtk.Label(label="Device ")
            self.device_combo = Gtk.ComboBoxText()

            self.generate_button = Gtk.Button(label="Generate")
            self.generate_button.set_size_request(100, -1)
            self.load_model_button = Gtk.Button(label="Load")
            hbox.pack_start(Gtk.Label(), True, True, 0)
            hbox.pack_start(device_label, False, False, 0)
            hbox.pack_start(self.device_combo, False, False, 0)
            hbox.pack_start(self.generate_button, False, False, 0)

            grid = Gtk.Grid()
            grid.set_column_homogeneous(False)
            grid.set_row_spacing(5)
            grid.set_column_spacing(10)
            grid.set_hexpand(False)

            model_label = Gtk.Label(label="OpenVINO Model")
            model_label.set_halign(Gtk.Align.START)
            model_label.set_valign(Gtk.Align.CENTER)
            self.model_combo = Gtk.ComboBoxText()
            self.model_combo.set_halign(Gtk.Align.START)
            self.model_combo.set_size_request(120, -1)

            hbox_model_settings = Gtk.Box(
                orientation=Gtk.Orientation.HORIZONTAL, spacing=6
            )
            hbox_model_settings.set_size_request(200, -1)
            hbox_model_settings.pack_start(hbox, False, False, 0)
            self.edit_models_btn = Gtk.Button(label="Edit")

            hbox_model_settings.pack_start(self.model_combo, False, False, 0)
            hbox_model_settings.pack_start(self.load_model_button, False, False, 0)
            hbox_model_settings.pack_start(self.edit_models_btn, False, False, 0)

            grid.attach(model_label, 0, 0, 1, 1)
            grid.attach(hbox_model_settings, 1, 0, 1, 1)

            inference_label = Gtk.Label(label="Number of inference steps")
            inference_label.set_halign(Gtk.Align.START)
            inference_label.set_valign(Gtk.Align.CENTER)

            self.inference_scale = Gtk.SpinButton()
            self.inference_scale.set_range(1, 50)
            self.inference_scale.set_value(4)
            self.inference_scale.set_increments(1, 10)
            self.inference_scale.set_input_purpose(Gtk.InputPurpose.NUMBER)

            grid.attach(inference_label, 0, 1, 1, 1)
            grid.attach(self.inference_scale, 1, 1, 1, 1)

            width_label = Gtk.Label(label="Image Width")
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

            height_label = Gtk.Label(label="Image Height")
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

            seed_label = Gtk.Label(label="Seed")
            seed_label.set_halign(Gtk.Align.START)
            seed_label.set_valign(Gtk.Align.CENTER)

            self.seed = Gtk.Entry.new()
            self.seed.set_placeholder_text("If left blank, random seed will be set..")

            grid.attach(seed_label, 0, 4, 1, 1)
            grid.attach(self.seed, 1, 4, 1, 1)

            guidance_label = Gtk.Label(label="Guidance Scale")
            guidance_label.set_halign(Gtk.Align.START)
            guidance_label.set_valign(Gtk.Align.CENTER)

            self.guidance_scale = Gtk.SpinButton()
            self.guidance_scale.set_range(0, 10.0)
            self.guidance_scale.set_value(1.0)
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
            self.update_ui()

            def on_device_changed(combo):
                device = combo.get_active_text()
                cur_device = self._get_current_device()
                if self.is_model_loaded and not self.is_model_changed:
                    self.generate_button.set_sensitive(device == cur_device)

            def on_model_changed(combo):
                self.is_model_changed = True
                model_name = self.get_model_full_path()
                cur_model = sd_option_cache.get("model_name")
                if self.is_model_loaded:
                    if model_name != cur_model:
                        self.generate_button.set_sensitive(False)
                    else:
                        self.generate_button.set_sensitive(True)
                self._update_devices_combo(model_name)
                self.is_model_changed = False

            self.device_combo.connect("changed", on_device_changed)
            self.model_combo.connect("changed", on_model_changed)

            if is_server_running():
                self.generate_button.set_sensitive(True)
                self.is_model_loaded = True
                current_model = sd_option_cache.get("model_name")
                self.device_combo.set_active(
                    self.find_index_by_text(
                        self.device_combo,
                        self._get_current_device(),
                    )
                )
                mindex = self.find_index_by_text(
                    self.model_combo,
                    current_model,
                    1,
                )
                self.model_combo.set_active(mindex)
            else:
                self.generate_button.set_sensitive(False)
                self.is_model_loaded = False

            def on_load_model_clicked(button):
                self.model_path = self.get_model_full_path()

                server = STABLE_DIFFUSION_OV_SERVER
                server_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    "openvino_utils",
                    "tools",
                    server,
                )
                selected_device = self.device_combo.get_active_text()
                self.set_sensitive(False)
                self.is_model_loaded = False
                run_load_model_thread = Thread(
                    target=self.async_load_models,
                    args=(
                        python_path,
                        server_path,
                        self.model_path,
                        [selected_device],
                        "",
                        True,
                        dialog,
                    ),
                )

                sd_option_cache.set("model_name", self.model_path)
                sd_option_cache.save()
                self.models_config.save(
                    "device_name", self.device_combo.get_active_text()
                )
                run_load_model_thread.start()

            def on_edit_models_clicked(button):
                models_before = self.models
                dlg = ModelManagerDialog()
                response = dlg.run()
                self._load_model_config()
                models_after = self.models

                if models_after != models_before:
                    msg_dlg = Gtk.MessageDialog(
                        parent=dialog,
                        flags=0,
                        message_type=Gtk.MessageType.ERROR,
                        buttons=Gtk.ButtonsType.OK,
                        text="FastSD - OpenVINO",
                        secondary_text="Info",
                    )
                    msg_dlg.format_secondary_text(
                        "Since models changed please reopen FastSD plugin"
                    )
                    msg_dlg.run()
                    msg_dlg.destroy()
                    dialog.destroy()

            def on_generate_clicked(button):
                if not is_server_running():
                    show_error_dialog(
                        dialog,
                        "Error : First load the model to generate images.",
                    )
                    return

                if self.seed.get_text() != "":
                    seed_val = self.seed.get_text()
                    if not self._is_valid_seed(seed_val):
                        show_error_dialog(
                            dialog,
                            "Error : Invalid seed value, please enter a number or leave seed blank for random seed",
                        )
                        return
                if "sana-sprint" in self.model_path:
                    inference_steps = int(self.inference_scale.get_value())
                    if inference_steps != 2:
                        show_error_dialog(
                            dialog,
                            "Error : SANA Sprint models needs 2 inference steps, please set it in the options.",
                        )
                        return

                self.update_gen_settings(sd_option_cache)

                runner = SDRunner(procedure, image, layer, self.config_path_output)
                run_inference_thread = Thread(
                    target=self._async_sd_run_func,
                    args=(
                        runner,
                        dialog,
                    ),
                )
                run_inference_thread.start()
                self.set_sensitive(False)
                self.generate_button.set_label("Generating...")

            self.generate_button.connect("clicked", on_generate_clicked)
            self.edit_models_btn.connect("clicked", on_edit_models_clicked)
            self.load_model_button.connect("clicked", on_load_model_clicked)

            dialog.get_content_area().add(vbox)
            dialog.show_all()
            response = dialog.run()
            if response == Gtk.ResponseType.HELP:
                url = "https://github.com/intel/openvino-ai-plugins-gimp/blob/main/README.md"
                Gio.app_info_launch_default_for_uri(url, None)
            dialog.destroy()

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


Gimp.main(FastSDPlugin.__gtype__, sys.argv)
