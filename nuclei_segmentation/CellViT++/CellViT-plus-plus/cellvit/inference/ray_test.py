# === test_ray_imports.py ===
import os
import sys
import ray
import time

# Get the current directory and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(project_root)
project_root = os.path.dirname(project_root)
sys.path.append(project_root)

# Logger function
def log_message(message, level="INFO"):
    """Log messages with timestamps and severity levels."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

# Helper function to check paths and imports in Ray workers
@ray.remote
def test_import():
    """Ray worker test for importing cellvit components."""
    import sys
    log_message("\nWorker sys.path:")
    for path in sys.path:
        log_message(f"  - {path}")
    
    try:
        from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
        log_message("\nSuccessfully imported cellvit in worker!")
        return "Success"
    except Exception as e:
        error_message = f"Failed to import cellvit in worker: {str(e)}"
        log_message(error_message, level="ERROR")
        return error_message

def main():
    """Main function to test cellvit imports."""
    status_code = 1  # Assume success

    # Print main process paths
    log_message("\nMain process sys.path:")
    for path in sys.path:
        log_message(f"  - {path}")

    # Initialize Ray with runtime environment
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": project_root  # Set PYTHONPATH for workers
        }
    }
    
    try:
        ray.init(runtime_env=runtime_env)
        log_message("\nRay initialized successfully.")
    except Exception as e:
        log_message(f"Error initializing Ray: {str(e)}", level="ERROR")
        status_code = 0  # Error initializing Ray
        return status_code

    # Test import in main process
    log_message("\nTesting import in main process:")
    try:
        from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
        log_message("Main process import successful.")
    except Exception as e:
        log_message(f"Main process import failed: {str(e)}", level="ERROR")
        status_code = 0  # Import failed in main process

    # Test import in Ray worker
    log_message("\nTesting import in Ray worker:")
    try:
        result = ray.get(test_import.remote())
        log_message(f"Worker result: {result}")
        if "Failed" in result:
            status_code = 0  # Import failed in worker
    except Exception as e:
        log_message(f"Ray worker test failed: {str(e)}", level="ERROR")
        status_code = 0  # Worker test failed

    # Shutdown Ray
    ray.shutdown()
    log_message("\nRay has been shut down.")

    return status_code

if __name__ == "__main__":
    status_code = main()
    print(status_code)
    # exit(status_code)  # Return status code (1 for success, 0 for errors)
