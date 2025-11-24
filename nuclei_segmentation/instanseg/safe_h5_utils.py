"""
Safe H5 file operations with retry logic to handle file locking issues.

This module provides utilities for safely opening H5 files with retry mechanisms
to handle Windows file locking errors that can occur when multiple processes
try to access the same H5 file simultaneously.
"""

import time
import h5py
from typing import Optional


def safe_open_h5_file(file_path: str, mode: str = 'r', max_retries: int = 5, retry_delay: float = 1.0, timeout: float = 30.0):
    """
    Safely open an H5 file with retry logic to handle file locking issues.

    Args:
        file_path: Path to the H5 file
        mode: File open mode ('r', 'r+', 'a', 'w')
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Maximum total time to wait in seconds

    Returns:
        h5py.File object or None if failed

    Raises:
        OSError: If file cannot be opened after all retries
    """
    start_time = time.time()

    for attempt in range(max_retries + 1):
        try:
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                raise OSError(f"Timeout waiting for file {file_path} to become available after {timeout}s")

            # Try to open the file
            h5_file = h5py.File(file_path, mode)
            print(f"[DEBUG] safe_open_h5_file - Successfully opened {file_path} on attempt {attempt + 1}")
            return h5_file

        except OSError as e:
            error_msg = str(e).lower()

            # Check if it's a file locking error
            if any(keyword in error_msg for keyword in ['lock', 'unable to synchronously open', 'errno = 0', 'getlasterror']):
                if attempt < max_retries:
                    elapsed = time.time() - start_time
                    print(f"[DEBUG] safe_open_h5_file - File {file_path} is locked (attempt {attempt + 1}/{max_retries + 1}), waiting {retry_delay}s (elapsed: {elapsed:.1f}s)")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[ERROR] safe_open_h5_file - Failed to open {file_path} after {max_retries + 1} attempts due to file locking")
                    raise OSError(f"Unable to open file {file_path} after {max_retries + 1} attempts due to file locking: {e}")
            else:
                # Not a locking error, re-raise immediately
                print(f"[ERROR] safe_open_h5_file - Failed to open {file_path} due to non-locking error: {e}")
                raise e
        except Exception as e:
            print(f"[ERROR] safe_open_h5_file - Unexpected error opening {file_path}: {e}")
            raise e

    # This should never be reached, but just in case
    raise OSError(f"Failed to open {file_path} after {max_retries + 1} attempts")


def safe_h5_context_manager(file_path: str, mode: str = 'r', max_retries: int = 5, retry_delay: float = 1.0, timeout: float = 30.0):
    """
    Context manager for safely opening H5 files with retry logic.

    Usage:
        with safe_h5_context_manager('file.h5', 'a') as hf:
            # Do operations with hf
            pass
    """
    class SafeH5Context:
        def __init__(self, file_path, mode, max_retries, retry_delay, timeout):
            self.file_path = file_path
            self.mode = mode
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            self.timeout = timeout
            self.h5_file = None

        def __enter__(self):
            self.h5_file = safe_open_h5_file(
                self.file_path,
                self.mode,
                self.max_retries,
                self.retry_delay,
                self.timeout
            )
            return self.h5_file

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.h5_file:
                try:
                    self.h5_file.close()
                except Exception as e:
                    print(f"[WARN] safe_h5_context_manager - Error closing file {self.file_path}: {e}")

    return SafeH5Context(file_path, mode, max_retries, retry_delay, timeout)


def safe_h5_open(file_path: str, mode: str = 'r', max_retries: int = 5, retry_delay: float = 1.0, timeout: float = 30.0):
    """
    Convenience function that returns a context manager for safe H5 file operations.

    This is a shorter alias for safe_h5_context_manager for easier usage.

    Usage:
        with safe_h5_open('file.h5', 'a') as hf:
            # Do operations with hf
            pass
    """
    return safe_h5_context_manager(file_path, mode, max_retries, retry_delay, timeout)
