# This is a PyInstaller runtime hook to fix multiprocessing issues on Windows
import os
import sys
import multiprocessing
import multiprocess

# Set multiprocessing start method
if sys.platform.startswith('win'):
    # Windows platform uses 'spawn' method instead of 'fork'
    multiprocessing.set_start_method('spawn', force=True)
    multiprocess.set_start_method('spawn', force=True)
    
    # Fix multiprocessing path issues on Windows
    if hasattr(sys, 'frozen'):
        module_dir = os.path.dirname(sys.executable)
        
        # Set _MEIPASS2 environment variable to help multiprocessing find the correct path
        os.environ["_MEIPASS2"] = sys._MEIPASS
        
        # Ensure multiprocessing can find necessary DLL files
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"] 