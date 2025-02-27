import os
import time
import random
import string
import socket
import subprocess
from typing import Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Global logging of sub-service information
SERVICE_REGISTRY: Dict[str, Dict] = {}

def get_random_env_name(prefix='child_env'):
    """Generate a random environment name such as child_env_20250126141530_abcd"""
    timestamp = time.strftime("%Y%m%d%H%M%S")
    rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{prefix}_{timestamp}_{rand_str}"

def find_free_port(start_port=8001, max_tries=100):
    """
    Starting with start_port,
    try the bindings in order,
    and return if you find an available port.
    """
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                port += 1
    return None

class CreateEnvRequest(BaseModel):
    env_prefix: Optional[str] = "fastapi_child"
    python_version: Optional[str] = "3.9"

@app.post("/create_env")
def create_env(payload: CreateEnvRequest):
    """
    Create a new Conda environment
    and install the sub-environment dependencies,
    then start the sub-service.
    """
    env_name = get_random_env_name(payload.env_prefix)
    python_ver = payload.python_version

    # 1. conda create
    create_env_cmd = f"conda create -n {env_name} python={python_ver} -y"
    try:
        subprocess.run(create_env_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return {"status": "fail", "message": f"Failed to create environment: {e}"}

    # 2. Install the child environment dependencies (requirements_child.txt in this case)
    # Assume requirements_child.txt and manager_service.py are in the same directory.
    install_cmd = f"conda run -n {env_name} python -m pip install -r requirements_child.txt"
    try:
        subprocess.run(install_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return {"status": "fail", "message": f"install dependencies fail: {e}"}

    # 3. Find an available port
    port = find_free_port(start_port=8001)
    if port is None:
        return {"status": "fail", "message": "not port available"}

    # 4. launch child service
    #    use python -m uvicorn
    service_cmd = f"conda run -n {env_name} python -m uvicorn separate_service:app --host 0.0.0.0 --port {port}"
    proc = subprocess.Popen(service_cmd, shell=True)

    # 5. record to manager
    SERVICE_REGISTRY[env_name] = {
        "port": port,
        "process": proc
    }

    return {
        "status": "success",
        "message": f"already created {env_name} and launch child service",
        "env_name": env_name,
        "port": port
    }

@app.get("/list_services")
def list_services():
    """
    list all services
    """
    result = {}
    for env_name, info in SERVICE_REGISTRY.items():
        result[env_name] = {
            "port": info["port"],
            "pid": info["process"].pid if info["process"] else None
        }
    return result

class StopServiceRequest(BaseModel):
    env_name: str

@app.post("/stop_service")
def stop_service(payload: StopServiceRequest):
    """
    stop and delete the service
    """
    env_name = payload.env_name
    if env_name not in SERVICE_REGISTRY:
        return {"status": "fail", "message": f"env {env_name} not exist"}

    proc = SERVICE_REGISTRY[env_name]["process"]
    if proc.poll() is None:  # still running
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()

    remove_env_cmd = f"conda env remove -n {env_name} -y"
    try:
        subprocess.run(remove_env_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return {"status": "fail", "message": f"delete env fail: {e}"}

    del SERVICE_REGISTRY[env_name]

    return {"status": "success", "message": f"env {env_name} has been stopped and deleted"}
