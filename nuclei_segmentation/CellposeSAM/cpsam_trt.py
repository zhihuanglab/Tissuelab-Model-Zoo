"""TensorRT path for Cellpose-SAM's network (ONNX Runtime TensorRT EP, bf16).

The SAM ViT-L encoder is ~85% of an eval. Run through TensorRT it is ~40% faster
than PyTorch bf16 with the same cells (measured on an H200: 3000x3000 IHC image,
6.6 s -> 4.1 s, 1,566 vs 1,567 cells, 99.8% of centroids within 1 px of fp32).
fp16 is not an option: it drops ~4% of the cells, in TensorRT as in PyTorch.

On first use the net is exported to ONNX (batch fixed at TRT_BATCH) and a bf16
engine is built (~90 s); both are cached under the cache directory, so later
processes only load them (~25 s, once per node process). Anything missing
(CUDA, onnxruntime-gpu, tensorrt) or failing leaves the caller on PyTorch.
CELLPOSE_SAM_TRT=0 turns the path off.
"""
import os
import sys
import time
from pathlib import Path

# A batch of 16 tiles already fills the GPU; larger ones are not faster but need
# far more memory (bs 64: 10 GB, and no engine at all within a 2 GB workspace).
TRT_BATCH = 16
TRT_WORKSPACE_GB = 2
ONNX_OPSET = 17
TILE = 256  # Cellpose's tile size (bsize)

# site-packages dirs ORT's TensorRT EP must find (pip tensorrt / CUDA wheels).
_LIB_SUBDIRS = (
    ("onnxruntime", "capi"),
    ("tensorrt_libs",),
    ("nvidia", "cudnn", "lib"),
    ("nvidia", "cublas", "lib"),
    ("nvidia", "cuda_runtime", "lib"),
)
_WIN_LIB_SUBDIRS = (
    ("onnxruntime", "capi"),
    ("tensorrt_libs",),
    ("nvidia", "cudnn", "bin"),
    ("nvidia", "cublas", "bin"),
    ("nvidia", "cuda_runtime", "bin"),
)


def enabled():
    return os.environ.get("CELLPOSE_SAM_TRT", "1") != "0"


def cache_dir():
    """CELLPOSE_SAM_CACHE, else cache/ next to the node (next to the executable when frozen)."""
    if os.environ.get("CELLPOSE_SAM_CACHE"):
        return Path(os.environ["CELLPOSE_SAM_CACHE"])
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    return base / "cache"


def _lib_dirs(subdirs):
    dirs = []
    for sp in sys.path:
        base = Path(sp)
        if base.name != "site-packages" or not base.is_dir():
            continue
        dirs.extend(str(base.joinpath(*sub)) for sub in subdirs if base.joinpath(*sub).is_dir())
    return list(dict.fromkeys(dirs))


def ensure_runtime_env():
    """Put the TensorRT / CUDA wheel libraries on the loader path. Call before
    torch or onnxruntime is imported.

    Linux reads LD_LIBRARY_PATH only at process start, and TissueLab may launch
    the node without these dirs; ORT then silently drops the TensorRT EP. So the
    process re-execs itself once with the path set (same PID, same arguments).
    """
    if not enabled():
        return
    if sys.platform == "win32":
        dirs = _lib_dirs(_WIN_LIB_SUBDIRS)
        if not any(d.endswith("tensorrt_libs") for d in dirs):
            return
        os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
        for d in dirs:
            os.add_dll_directory(d)
        return
    dirs = _lib_dirs(_LIB_SUBDIRS)
    if not any(d.endswith("tensorrt_libs") for d in dirs):
        return  # TensorRT not installed: PyTorch it is
    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    missing = [d for d in dirs if d not in current]
    if not missing or os.environ.get("CELLPOSE_SAM_TRT_REEXEC") == "1":
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + current)
    os.environ["CELLPOSE_SAM_TRT_REEXEC"] = "1"
    # orig_argv keeps the interpreter flags (-u, -I, ...); a frozen build's is [exe, args...].
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable] + sys.argv)
    # Python finds its venv from argv[0]; a relative one breaks once the cwd moved.
    argv[0] = sys.executable
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if not getattr(sys, "frozen", False) and main_file and sys.argv and sys.argv[0] in argv:
        # The script path may be relative to a directory the process has since left.
        argv[argv.index(sys.argv[0])] = os.path.abspath(main_file)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, argv)


class TrtNet:
    """Stands in for CellposeModel.net. cellpose.core calls it with a float batch
    of n <= TRT_BATCH tiles on the GPU and reads the first output (flows)."""

    def __init__(self, session, orig_net):
        import torch
        self._session = session
        self._orig = orig_net  # kept (on the CPU) for attribute lookups such as diam_mean
        self.dtype = torch.float32
        self.device = torch.device("cuda")
        self.batch_size = TRT_BATCH

    def __getattr__(self, name):
        return getattr(self._orig, name)

    def eval(self):
        return self

    def __call__(self, X):
        import numpy as np
        import torch
        n = X.shape[0]
        Xp = X.float()
        if n < TRT_BATCH:
            Xp = torch.cat([Xp, Xp.new_zeros((TRT_BATCH - n, *Xp.shape[1:]))])
        Xp = Xp.contiguous()
        out = torch.empty((TRT_BATCH, 3, TILE, TILE), device=X.device, dtype=torch.float32)
        io = self._session.io_binding()
        io.bind_input("input", "cuda", 0, np.float32, tuple(Xp.shape), Xp.data_ptr())
        io.bind_output("flows", "cuda", 0, np.float32, tuple(out.shape), out.data_ptr())
        torch.cuda.synchronize()  # ORT runs on its own stream
        self._session.run_with_iobinding(io)
        return out[:n], torch.zeros((n, 256), device=X.device)


def _export_onnx(make_fp32_net, path):
    import torch

    class Flows(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[0]

    net = make_fp32_net().eval()
    net.requires_grad_(False)
    x = torch.zeros(TRT_BATCH, 3, TILE, TILE, device="cuda")
    tmp = path.with_suffix(".onnx.tmp")
    with torch.no_grad():
        torch.onnx.export(Flows(net), (x,), str(tmp), input_names=["input"], output_names=["flows"],
                          opset_version=ONNX_OPSET, dynamo=False, do_constant_folding=True)
    os.replace(tmp, path)
    del net
    torch.cuda.empty_cache()


def try_build(orig_net, make_fp32_net, log=print):
    """A TrtNet for Cellpose-SAM, or None when the TensorRT path is unavailable.

    orig_net: the model's PyTorch net (moved to the CPU when TensorRT takes over).
    make_fp32_net: builds the net in float32, used once to export the ONNX.
    """
    if not enabled():
        log("[Cellpose-SAM] TensorRT disabled (CELLPOSE_SAM_TRT=0); using PyTorch")
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        import onnxruntime as ort
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            log("[Cellpose-SAM] onnxruntime has no TensorRT EP; using PyTorch")
            return None
        root = cache_dir()
        root.mkdir(parents=True, exist_ok=True)
        onnx_path = root / f"cpsam_v2_b{TRT_BATCH}.onnx"
        if not onnx_path.is_file():
            t = time.time()
            _export_onnx(make_fp32_net, onnx_path)
            log(f"[Cellpose-SAM] exported {onnx_path} in {time.time() - t:.0f}s")
        engines = root / "trt_engines"
        engines.mkdir(exist_ok=True)
        opts = {
            "device_id": torch.cuda.current_device(),
            "trt_bf16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(engines),
            "trt_max_workspace_size": str(TRT_WORKSPACE_GB << 30),
            "trt_builder_optimization_level": 3,
        }
        so = ort.SessionOptions()
        so.log_severity_level = 3
        t = time.time()
        session = ort.InferenceSession(str(onnx_path), so,
                                       providers=[("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"])
        if session.get_providers()[0] != "TensorrtExecutionProvider":
            log("[Cellpose-SAM] TensorRT EP did not load (libraries missing?); using PyTorch")
            return None
        trt_net = TrtNet(session, orig_net)
        trt_net(torch.zeros((1, 3, TILE, TILE), device="cuda"))  # build or load the engine now, not mid-slide
        log(f"[Cellpose-SAM] TensorRT bf16 engine ready in {time.time() - t:.0f}s (batch {TRT_BATCH})")
        trt_net._orig = orig_net.cpu()
        torch.cuda.empty_cache()
        return trt_net
    except Exception as e:  # any failure: stay on PyTorch rather than fail the run
        log(f"[Cellpose-SAM] TensorRT unavailable ({type(e).__name__}: {e}); using PyTorch")
        return None
