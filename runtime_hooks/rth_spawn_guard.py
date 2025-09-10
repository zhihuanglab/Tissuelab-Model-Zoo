# Runtime hook for PyInstaller: guard spawn helpers and enable freeze_support
import sys
import multiprocessing as _mp

# Ensure frozen spawn mode works without CLI re-entry issues
try:
    _mp.freeze_support()
except Exception:
    pass

# If this process is a multiprocessing resource_tracker helper, exit early
try:
    _argv = " ".join(sys.argv)
    if "multiprocessing.resource_tracker" in _argv:
        sys.exit(0)
except Exception:
    pass


