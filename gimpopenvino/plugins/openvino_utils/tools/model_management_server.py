import os
import json
import sys
import socket
import ast
import traceback
import logging as log
from pathlib import Path
import psutil
import threading

sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common")])
sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","openvino_utils","tools")])

from tools_utils import get_weight_path
from model_manager import ModelManager

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65434  # Port to listen on (stable_diffusion_ov_server uses port 65432 &  65433)

# This function is run on a dedicated thread when a new connection is established.
def run_connection_routine(model_manager, conn):
    with conn:
        while True:
            data = conn.recv(1024)

            if not data:
                break

            if data.decode() == "kill":
                os._exit(0)
            if data.decode() == "ping":
                conn.sendall(data)
                continue

            # request to get the details and state of all supported models
            if data.decode() == "get_all_model_details":

                # get the list of installed models, and installable model details.
                installed_models, installable_model_details = model_manager.get_all_model_details()

                # Send the list of installed models
                num_installed_models = len(installed_models)
                conn.sendall(bytes(str(num_installed_models), 'utf-8'))
                data = conn.recv(1024) # <-wait for ack
                for i in range(0, num_installed_models):
                    for detail in ["name", "id"]:
                        conn.sendall(bytes(installed_models[i][detail], 'utf-8'))
                        data = conn.recv(1024) # <-wait for ack

                # Send the installable model details
                num_installable_models = len(installable_model_details)
                conn.sendall(bytes(str(num_installable_models), 'utf-8'))
                data = conn.recv(1024) # <-wait for ack
                for i in range(0, num_installable_models):
                    for detail in ["name", "description", "id", "install_status"]:
                        conn.sendall(bytes(installable_model_details[i][detail], 'utf-8'))
                        data = conn.recv(1024) # <-wait for ack

                continue

            # request to install a model.
            if data.decode() == "install_model":
               #send ack
               conn.sendall(data)

               #get model id.
               #TODO: Need a timeout here.
               model_id = conn.recv(1024).decode()

               if model_id not in model_manager.model_install_status:

                   # add it to the dictionary here (instead of in model_manager.install_model).
                   # This will guarantee that it is present in the dictionary before sending the ack,
                   #  and avoiding a potential race condition where the GIMP UI side asks for the status
                   #  before the thread spawns.
                   model_manager.model_install_status[model_id] = {"status": "Preparing to install..", "percent": 0.0}

                   #Run the install on another thread. This allows the server to service other requests
                   # while the install is taking place.
                   install_thread = threading.Thread(target=model_manager.install_model, args=(model_id,))
                   install_thread.start()
               else:
                   print(model_id, "is already currently installing!")

               #send ack
               conn.sendall(data)

               continue

            # request to get the status of a model that is getting installed.
            if data.decode() == "install_status":

                # send ack
                conn.sendall(data)

                # make a copy of this so that the number of entries doesn't change while we're
                #  in this routine.
                model_install_status = model_manager.model_install_status.copy()

                # Get the model-id that we are interested in.
                data = conn.recv(1024)
                model_id = data.decode()

                if model_id in model_install_status:
                    details = model_install_status[model_id]

                    status = details["status"]
                    perc = details["percent"]
                else:
                    # the model_id is not found in the installer map... set status to "done" / 100.0
                    # TODO: What about failure cases?
                    status = "done"
                    perc = 100.0

                # first, send the status
                conn.sendall(bytes(status, 'utf-8'))
                data = conn.recv(1024) # <- get ack

                # then, send the send the percent
                conn.sendall(bytes(str(perc), 'utf-8'))
                data = conn.recv(1024) # <- get ack

                continue

            if data.decode() == "error_details":
                # send ack
                conn.sendall(data)

                # Get the model-id that we are interested in.
                data = conn.recv(1024)
                model_id = data.decode()

                summary, details = model_manager.get_error_details(model_id)

                # first, send the summary
                conn.sendall(bytes(summary, 'utf-8'))
                data = conn.recv(1024) # <- get ack

                # then, send the send the details
                conn.sendall(bytes(details, 'utf-8'))
                data = conn.recv(1024) # <- get ack


                continue

            if data.decode() == "install_cancel":
                # send ack
                conn.sendall(data)

                # Get the model-id that we are interested in.
                data = conn.recv(1024)
                model_id = data.decode()

                #send ack
                conn.sendall(data)

                model_manager.cancel_install(model_id)

                continue

            print("Warning! Unsupported command sent: ", data.decode())

def run():
    weight_path = get_weight_path()

    #Move to a temporary working directory in a known place.
    # This is where we'll be downloading stuff to, etc.
    tmp_working_dir=os.path.join(weight_path, '..', 'mms_tmp')

    #if this dir doesn't exist, create it.
    if not os.path.isdir(tmp_working_dir):
        os.mkdir(tmp_working_dir)

    # go there.
    os.chdir(tmp_working_dir)

    model_manager = ModelManager(weight_path)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        while True:
            conn, addr = s.accept()

            #Run this connection on a dedicated thread. This allows multiple connections to be present at once.
            conn_thread = threading.Thread(target=run_connection_routine, args=(model_manager, conn))
            conn_thread.start()



def start():

    run_thread = threading.Thread(target=run, args=())
    run_thread.start()

    gimp_proc = None
    for proc in psutil.process_iter():
        if "gimp-2.99" in proc.name():
            gimp_proc = proc
            break

    if gimp_proc:
        psutil.wait_procs([proc])
        os._exit(0)

    run_thread.join()

if __name__ == "__main__":
   start()
