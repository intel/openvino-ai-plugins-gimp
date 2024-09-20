
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, Gdk
import threading
import time
import sys
import os
import socket
import subprocess

class ErrorWindow(Gtk.Dialog):
    def __init__(self, parent, summary, details):
        Gtk.Dialog.__init__(self, title="Error", transient_for=parent, flags=0)
        self.set_default_size(1200, 800)

        # Add buttons (OK button to close the dialog)
        self.add_button(Gtk.STOCK_OK, Gtk.ResponseType.OK)

        # Create a box to contain the content
        box = self.get_content_area()

        # Add margins to the content area
        box.set_margin_top(20)
        box.set_margin_bottom(20)
        box.set_margin_start(20)
        box.set_margin_end(20)

        # High-level summary
        summary_label = Gtk.Label(label=summary)
        summary_label.set_xalign(0)  # Align text to the left
        summary_label.set_margin_bottom(10)
        summary_label.set_markup(f"<b>{summary}</b>")  # Bold the summary text
        box.add(summary_label)

        # Detailed description using Gtk.TextView inside a Gtk.ScrolledWindow
        details_view = Gtk.TextView()
        details_view.set_wrap_mode(Gtk.WrapMode.WORD)
        details_view.set_editable(False)
        details_view.set_cursor_visible(False)
        text_buffer = details_view.get_buffer()
        text_buffer.set_text(details)

        # Add the TextView to a ScrolledWindow
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_vexpand(True)
        scrolled_window.add(details_view)
        box.add(scrolled_window)

        # Show all widgets
        self.show_all()

class ModelManagementWindow(Gtk.Window):
    def __init__(self, config_path, python_path, models_updated_callback):
        Gtk.Window.__init__(self, title="Stable Diffusion Model Management")
        #self.set_default_size(1200, 800)

        #self.hide()
        self.set_visible(False)

        self._models_updated_callback = models_updated_callback
        self._host = "127.0.0.1"
        self._port = 65434
        server = "model_management_server.py"
        server_path = os.path.join(config_path, server)

        #if it's not running already, start it up!
        if( self.is_server_running() is False ):
            _process = subprocess.Popen([python_path, server_path], close_fds=True)

        # make sure that the server is running before we proceed...
        connect_retries = 5
        while self.is_server_running() is False:
           time.sleep(1)
           connect_retries = connect_retries - 1

           if connect_retries<=0:
               raise RuntimeError("Error in Model Management Server startup...")

        # Connect the delete-event signal to a custom handler
        self.connect("delete-event", self.on_delete_event)

        self._installed_models, installable_model_details = self.get_all_model_details()

        self.poll_install_status_thread = None
        self.bStopPoll = False

        model_box = Gtk.Grid()
        model_box.set_row_spacing(20)
        model_box.set_column_spacing(20)
        model_box.set_margin_start(20)
        model_box.set_margin_end(20)
        model_box.set_margin_top(20)
        model_box.set_margin_bottom(20)
        self.add(model_box)

        model_row_index = 0

        self.model_box = model_box
        self.model_ui_map = {}

        self.expanded_descriptions = []

        # Add each model to the window
        if installable_model_details is not None:
            for model in installable_model_details:

                # Model name label
                name_label = Gtk.Label()
                name_label.set_markup("<b>" + model["name"] + "</b>")
                name_label.set_xalign(0)  # Align left
                name_label.set_name("model-name")
                name_label.set_sensitive(True)  # Make it sensitive to events
                name_label.set_visible(True)

                name_label.set_halign(Gtk.Align.START)
                name_label.set_valign(Gtk.Align.CENTER)
                #title_event_box = Gtk.EventBox()
                #title_event_box.add(name_label)
                #title_event_box.connect("button-press-event", self.on_title_clicked, model_row_index)

                # Change cursor to hand pointer when hovering over the title
                #title_event_box.connect("enter-notify-event", self.on_mouse_enter)
                #title_event_box.connect("leave-notify-event", self.on_mouse_leave)

                model_box.attach(name_label, 0, model_row_index * 2, 1, 1)

                # Download button
                if model["install_status"] == "not_installed":
                    download_button = Gtk.Button(label="Install")
                    download_button.set_sensitive(True)
                elif model["install_status"] == "installed":
                    download_button = Gtk.Button(label="Installed")
                    download_button.set_sensitive(False)
                elif model["install_status"] == "installed_updates_available":
                    download_button = Gtk.Button(label="Update")
                    download_button.set_sensitive(True)
                elif model["install_status"] == "installing" or model["install_status"] == "install_error":
                    download_button = Gtk.Button(label="Install")
                    download_button.set_sensitive(False)

                download_button.connect("clicked", self.on_download_clicked, model["id"])

                # Attach the download button to the grid
                model_box.attach(download_button, 2, model_row_index * 2, 1, 1)

                self.model_ui_map[model["id"]] = { "row_index": model_row_index*2, "download_button": download_button}

                # if the state of the model is installing, kick off the poll thread for it.
                if model["install_status"] == "installing" or model["install_status"] == "install_error":
                    poll_install_status_thread = threading.Thread(target=self.poll_install_status, args=(model["id"],))
                    poll_install_status_thread.start()


                model_row_index += 1

        else:
            print("list of supported models is empty!")

        #self.show_all()

    def get_installed_model_list(self):
        return self._installed_models

    def on_title_clicked(self, widget, event, index):
         # Toggle the visibility of the detailed description
        detailed_description = self.expanded_descriptions[index]
        detailed_description.set_visible(not detailed_description.get_visible())

    def on_mouse_enter(self, widget, event):
        # Change the cursor to a hand pointer when the mouse enters the widget
        Gdk.Window.set_cursor(widget.get_window(), Gdk.Cursor.new_from_name(Gdk.Display.get_default(), "pointer"))

    def on_mouse_leave(self, widget, event):
        # Restore the default cursor when the mouse leaves the widget
        Gdk.Window.set_cursor(widget.get_window(), None)

    def display(self):
        self.show_all()
        for description in self.expanded_descriptions:
            description.set_visible(False)

    def __del__(self):
        # stop the polling thread.
        self.stop_poll_thread()

    def stop_poll_thread(self):
        self.bStopPoll = True

    def post_install_routine(self, model_id, install_status, error_summary, error_details):
        model_ui = self.model_ui_map[model_id]

        download_button = model_ui["download_button"]

        if "progress_bar" in model_ui:
            progress_bar = model_ui["progress_bar"]
            self.model_box.remove(progress_bar)
            model_ui.pop("progress_bar")

        if "cancel_button" in model_ui:
            cancel_button = model_ui["cancel_button"]
            self.model_box.remove(cancel_button)
            model_ui.pop("cancel_button")

        model_row_index = model_ui["row_index"]

        self.model_box.attach(download_button, 2, model_row_index, 1, 1)

        if install_status == "installed":
            download_button.set_label("Installed")
            download_button.set_sensitive(False)
        elif install_status == "installed_updates_available":
            download_button.set_label("Update")
            download_button.set_sensitive(True)
        elif install_status == "install_error":
            download_button.set_label("Install")
            download_button.set_sensitive(True)

            # Create and run the error dialog
            dialog = ErrorWindow(self, error_summary, error_details)
            dialog.run()
            dialog.destroy()
        else:
            download_button.set_label("Install")
            download_button.set_sensitive(True)

        download_button.set_visible(True)

        if self._models_updated_callback:
            self._models_updated_callback(self._installed_models)

    def update_ui_install_progress(self, model_id, install_status):

        model_ui = self.model_ui_map[model_id]

        model_row_index = model_ui["row_index"]

        status = install_status["status"]
        perc = install_status["percent"]

        if "progress_bar" not in model_ui:
            download_button = model_ui["download_button"]
            self.model_box.remove(download_button)

            progress_bar = Gtk.ProgressBar()
            progress_bar.set_show_text(True)
            progress_bar.set_visible(True)

            self.model_box.attach(progress_bar, 2, model_row_index, 1, 1)

            cancel_button = Gtk.Button(label="Cancel")
            cancel_button.set_visible(True)
            cancel_button.set_halign(Gtk.Align.CENTER)  # Center the button horizontally
            cancel_button.set_valign(Gtk.Align.CENTER)  # Center the button vertically
            self.model_box.attach(cancel_button, 2, model_row_index + 1, 1, 1)
            #cancel_button.connect("size-allocate", self.on_cancel_button_size_allocate)
            cancel_button.connect("clicked", self.on_cancel_clicked, model_id)

            model_ui["progress_bar"] = progress_bar
            model_ui["cancel_button"] = cancel_button
        else:
            progress_bar = model_ui["progress_bar"]

        progress_bar.set_text(status)
        progress_bar.set_fraction(perc / 100.0)


    def poll_install_status(self, model_id):

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self._host, self._port))

                install_status = self.get_install_status(s, model_id)

                while( install_status["status"] != "done" ):
                    GLib.idle_add(self.update_ui_install_progress, model_id, install_status)

                    # poll every 1 second.
                    time.sleep(1)

                    if self.bStopPoll is True:
                        break

                    install_status = self.get_install_status(s, model_id)

                # If we're here, the model installation is complete (or perhaps it failed)

                # We will now get details of all models
                self._installed_models, installable_model_details = self._all_model_details(s)

                for model_detail in installable_model_details:
                    # if this is the model that we are interested in..
                    if model_detail["id"] == model_id:
                        install_status = model_detail["install_status"]

                        error_summary = "None"
                        error_details = "None"

                        if install_status == "install_error":
                            summary, error_details = self.get_error_details(s, model_id)
                            error_summary = "Error during Installation of " + model_detail["name"] + "!"
                            error_summary += "\nSummary: " + summary
                            error_summary += "\n\nDetails:"


                        # launch the post_install routine (UI updates) on the main thread
                        GLib.idle_add(self.post_install_routine, model_id, install_status, error_summary, error_details)
                        break

        except Exception as e:
            print(f"There was a problem polling install status..")
            print(e)
            import traceback
            traceback.print_exc()



    def get_error_details(self, s, model_id):

        #send cmd
        s.sendall(b"error_details")

        #wait for an ack
        data = s.recv(1024)

        # send the model_id
        s.sendall(bytes(model_id, 'utf-8'))

        # get the summary
        data = s.recv(1024)
        summary = data.decode()
        s.sendall(data) # <- send ack

        # get the details
        data = s.recv(4096)
        details = data.decode()
        s.sendall(data) # <- send ack

        return summary, details

    def get_install_status(self, s, model_id):

        #send cmd
        s.sendall(b"install_status")

        #wait for an ack
        data = s.recv(1024)

        # send the model_id
        s.sendall(bytes(model_id, 'utf-8'))

        # get the status
        data = s.recv(1024)
        status = data.decode()
        s.sendall(data) # <- send ack

        # get the percent
        data = s.recv(1024)
        percent = float(data.decode())
        s.sendall(data) # <- send ack

        install_status = {"status": status, "percent": percent}

        return install_status


    def _all_model_details(self, s):
        #send cmd
        s.sendall(b"get_all_model_details")

        # get number of installed models
        data = s.recv(1024)
        num_models = int(data.decode())

        #send ack
        s.sendall(data)

        installed_models = []
        for i in range(0, num_models):
            model_detail = {}
            for detail in ["name", "id"]:
                data = s.recv(1024)
                model_detail[detail] = data.decode()
                #send ack
                s.sendall(data)

            installed_models.append(model_detail)

        # get the number of installable models
        data = s.recv(1024)
        num_installable_models = int(data.decode())

        #send ack
        s.sendall(data)

        installable_model_details = []
        for i in range(0, num_installable_models):
            model_detail = {}
            for detail in ["name", "description", "id", "install_status"]:
                data = s.recv(1024)
                model_detail[detail] = data.decode()
                #send ack
                s.sendall(data)

            installable_model_details.append(model_detail)

        return installed_models, installable_model_details

    def get_all_model_details(self):

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self._host, self._port))

                return self._all_model_details(s)

        except Exception as e:
            print(f"There was a problem getting model details..")
            print(e)


    def on_cancel_clicked(self, button, model_id):
        # prevent user from clicking it again while we kick off the cancel
        button.set_sensitive(False)

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self._host, self._port))

                #send cmd
                s.sendall(b"install_cancel")

                #wait for ack
                data = s.recv(1024)

                #send model name
                s.sendall(bytes(model_id, 'utf-8'))

                #wait for ack
                data = s.recv(1024)


        except Exception as e:
            print(f"There was a problem triggering cancel for {model_id}...")
            print(e)

    def on_download_clicked(self, button, model_id):

        # prevent user from clicking it again while we kick off the install
        button.set_sensitive(False)

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self._host, self._port))

                #send cmd
                s.sendall(b"install_model")

                #wait for ack
                data = s.recv(1024)

                #send model name
                s.sendall(bytes(model_id, 'utf-8'))

                #wait for ack
                data = s.recv(1024)

                poll_install_status_thread = threading.Thread(target=self.poll_install_status, args=(model_id,))
                poll_install_status_thread.start()

        except Exception as e:
            print(f"There was a problem downloading {model_id}...")
            print(e)

    def on_delete_event(self, widget, event):
        # Hide the window instead of destroying it
        self.hide()
        # Returning True prevents the window from being destroyed
        return True

    def is_server_running(self):
        ret = False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1) # <- set connection timeout to 100 ms (default is a few seconds)
                s.connect((self._host, self._port))
                s.sendall(b"ping")
                data = s.recv(1024)
                if data.decode() == "ping":
                    ret = True
        except Exception as e:
            ret = False
        return ret

