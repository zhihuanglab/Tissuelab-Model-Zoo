"""
Safe H5 file operations with proper locking and error handling
"""
import h5py
import fcntl
import time
import os

def safe_h5_open(path, mode='r', max_retries=10, retry_delay=0.5):
    """
    Safely open an H5 file with file locking to prevent concurrent access issues
    
    Args:
        path: Path to the H5 file
        mode: File mode ('r', 'r+', 'w', 'a', etc.)
        max_retries: Maximum number of retries if file is locked
        retry_delay: Delay between retries in seconds
    
    Returns:
        h5py.File object
    """
    for attempt in range(max_retries):
        try:
            # Open the H5 file
            h5_file = h5py.File(path, mode, libver='latest')
            
            # Try to acquire an exclusive lock (for write modes) or shared lock (for read mode)
            lock_type = fcntl.LOCK_EX if mode in ['w', 'a', 'r+'] else fcntl.LOCK_SH
            
            try:
                fcntl.flock(h5_file.id.get_vfd_handle(), lock_type | fcntl.LOCK_NB)
            except (IOError, OSError):
                # File is locked, close and retry
                h5_file.close()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise IOError(f"Could not acquire lock on {path} after {max_retries} attempts")
            
            return h5_file
            
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                raise IOError(f"Could not open {path} after {max_retries} attempts: {e}")
    
    raise IOError(f"Failed to open {path} after {max_retries} attempts")
