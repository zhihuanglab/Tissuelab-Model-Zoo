# -*- coding: utf-8 -*-
import time
import importlib.util
import subprocess

def check_module(module_name):
    """Check if a module is installed in the environment."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"Error: {module_name} is not installed.")
        return False
    print(f"{module_name} is installed.")
    return True


def log_message(message, level="INFO"):
    """Log messages with timestamps and severity levels."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def run_ray_test():
    """Run the ray_test.py script."""
    try:
        log_message("Executing 'cellvit/inference/ray_test.py'...", "INFO")
        result = subprocess.run(
            ["python", "cellvit/inference/ray_test.py"],
            check=True,
            capture_output=True,
            text=True
        )
        log_message(f"Script output:\n{result.stdout}", "SUCCESS")
    except subprocess.CalledProcessError as e:
        log_message(f"Error while running 'ray_test.py': {e.stderr}", "ERROR")
        raise e
    except Exception as e:
        log_message(f"Unexpected error while running 'ray_test.py': {e}", "ERROR")
        raise e

try:
    # Import essential libraries
    log_message("Checking installed libraries...")
    required_modules = ["torch", "torchaudio", "torchvision"]

    for module in required_modules:
        check_module(module)

    local_modules = [
        "cellvit.training.base_ml.base_cli",
        "cellvit.training.experiments.experiment_cell_classifier",
        "cellvit.inference.cli",
        "cellvit.inference.inference_memory",
    ]

    for module in local_modules:
        check_module(module)

    log_message("Environment is correctly set up.", "SUCCESS")
except ImportError as e:
    log_message(f"Error: {e}", "ERROR")
    raise e

# GPU availability check
try:
    import torch

    log_message("Checking GPU availability...")
    use_cuda = torch.cuda.is_available()

    if not use_cuda:
        raise SystemError("No CUDA-capable GPU detected.")

    log_message("CUDA-capable GPU detected.", "SUCCESS")
    log_message(f"CUDNN Version: {torch.backends.cudnn.version()}", "INFO")
    log_message(f"Number of CUDA Devices: {torch.cuda.device_count()}", "INFO")
    log_message(f"CUDA Device Name: {torch.cuda.get_device_name(0)}", "INFO")
    log_message(
        f"CUDA Device Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
        "INFO",
    )
except SystemError as e:
    log_message(f"GPU Error: {e}", "ERROR")
    raise e
except Exception as e:
    log_message(f"Unexpected error during GPU check: {e}", "ERROR")
    raise e

# CuPy availability and GPU access check
try:
    log_message("Checking CuPy availability...")
    import cupy as cp

    # Perform a simple CuPy operation to ensure it's functioning
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    c = a + b

    # Check if the operation was successful
    if cp.allclose(c, [5, 7, 9]):
        log_message("CuPy is functioning correctly.", "SUCCESS")
    else:
        raise RuntimeError("CuPy operation validation failed.")

    # Check if CuPy can access the GPU
    log_message("Checking GPU availability with CuPy...")
    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count == 0:
        raise SystemError("No CUDA-capable GPU detected by CuPy.")

    log_message(f"CuPy detected {device_count} CUDA device(s).", "SUCCESS")

    for device_id in range(device_count):
        # Set the current device
        cp.cuda.Device(device_id).use()

        # Get device properties
        props = cp.cuda.runtime.getDeviceProperties(device_id)

        log_message(f"Device ID: {device_id}")
        log_message(f"  Name: {props['name']}")
        log_message(f"  Total Global Memory: {props['totalGlobalMem']} bytes")
        log_message(f"  Multiprocessor Count: {props['multiProcessorCount']}")
        log_message(f"  Compute Capability: {props['major']}.{props['minor']}")
        log_message("")

    log_message("CuPy is able to access the GPU and perform operations.", "SUCCESS")
except ImportError as e:
    log_message(f"CuPy Import Error: {e}", "ERROR")
    raise e
except RuntimeError as e:
    log_message(f"CuPy Error: {e}", "ERROR")
    raise e
except SystemError as e:
    log_message(f"CuPy GPU Error: {e}", "ERROR")
    raise e
except Exception as e:
    log_message(f"Unexpected error during CuPy check: {e}", "ERROR")
    raise e

# CuCIM
try:
    log_message("Checking CuCIM availability...")
    from cucim import CuImage
    from cellvit.utils.download_example_files import check_test_database

    log_message("Downloading example files...")
    check_test_database()
    log_message("Opening example Image with CuCIM")
    image = CuImage("./test_database/x40_svs/JP2K-33003-2.svs")
    image.size()
    image.resolutions
    image.read_region((0, 0), (1000, 1000))
    log_message("Imported CuCIM and loaded example WSI", "SUCCESS")

except Exception as e:
    log_message(f"Unexpected error during CuCIM check: {e}", "ERROR")
    raise e

run_ray_test()

log_message("")
log_message("")
log_message(f"{60*'*'}")
log_message("")
log_message("")
log_message("Everything checked", "SUCCESS")
log_message("")
log_message("")
log_message(f"{60*'*'}")
