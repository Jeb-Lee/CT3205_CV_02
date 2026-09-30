"""Print the Python / library versions in use (paste this output into the report's appendix)."""
import importlib
import sys

print("python", sys.version.split()[0])
for m in ["cv2", "numpy", "scipy", "matplotlib", "open3d", "pandas", "tqdm", "PIL", "torch"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:11s}", getattr(mod, "__version__", "?"))
    except Exception:
        print(f"{m:11s}", "not installed")
try:
    import cv2
    print("opencv CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0)
except Exception:
    pass
try:
    import torch
    print("torch CUDA available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception:
    pass
