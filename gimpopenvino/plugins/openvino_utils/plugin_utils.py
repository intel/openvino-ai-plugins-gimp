# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

import gi
gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gimp, GimpUi, GObject, GLib, Gio, Gtk
import gettext

_ = gettext.gettext


def show_dialog(message, title, icon="logo", image_paths=None):
    use_header_bar = Gtk.Settings.get_default().get_property("gtk-dialogs-use-header")
    dialog = GimpUi.Dialog(use_header_bar=use_header_bar, title=_(title))
    # Add buttons
    dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
    dialog.add_button("_OK", Gtk.ResponseType.APPLY)
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, homogeneous=False, spacing=10)
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

    # Show Logo
    logo = Gtk.Image.new_from_file(image_paths[icon])
    # vbox.pack_start(logo, False, False, 1)
    grid.attach(logo, 0, 0, 1, 1)
    logo.show()
    # Show message
    label = Gtk.Label(label=_(message))
    # vbox.pack_start(label, False, False, 1)
    grid.attach(label, 1, 0, 1, 1)
    label.show()
    dialog.show()
    dialog.run()
    return


def save_image(image, drawable, file_path):
    interlace, compression = 0, 2
    pdb_proc   = Gimp.get_pdb().lookup_procedure('file-png-export')
    pdb_config = pdb_proc.create_config()
    pdb_config.set_property('run-mode', Gimp.RunMode.NONINTERACTIVE)
    pdb_config.set_property('image', image)
    pdb_config.set_property('file', Gio.File.new_for_path(file_path))
    pdb_config.set_property('options', None)
    pdb_config.set_property('interlaced', interlace)
    pdb_config.set_property('compression', compression)
    # write all PNG chunks except oFFs(ets)
    pdb_config.set_property('bkgd', True)
    pdb_config.set_property('offs', False)
    pdb_config.set_property('phys', True)
    pdb_config.set_property('time', True)
    pdb_config.set_property('save-transparent', True)
    pdb_proc.run(pdb_config)


def N_(message):
    return message