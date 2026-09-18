"""Trace the order of operations inside ``torch.load`` and ``load_state_dict``.

Run:  python pytorch/scripts/trace_load.py

Wraps the functions that the load-path flow chart in ../README.md names and
prints them in the order they actually run for a state_dict checkpoint, with
weights_only=True (the default). Consecutive repeats are collapsed.

Only the standard library and torch are needed.
"""

import functools
import io
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch._weights_only_unpickler as wo
import torch.serialization as ser

events: list[str] = []
depth = 0


def log(msg):
    events.append("  " * depth + msg)


def traced(name, fmt=None):
    """Decorator: log a call (with an optional argument summary) and nest what happens inside."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            global depth
            log(name + (f"  {fmt(*args, **kwargs)}" if fmt else ""))
            depth += 1
            try:
                return fn(*args, **kwargs)
            finally:
                depth -= 1

        return wrapper

    return deco


class ReaderProxy:
    """Stand-in for torch._C.PyTorchFileReader that logs every record access."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            arg = repr(args[0]) if args else ""
            log(f"PyTorchFileReader.{name}({arg})")
            return attr(*args, **kwargs)

        return call


def install():
    ser._open_file_like = traced("_open_file_like")(ser._open_file_like)
    ser._is_zipfile = traced("_is_zipfile")(ser._is_zipfile)
    ser._is_torchscript_zip = traced("_is_torchscript_zip")(ser._is_torchscript_zip)

    orig_reader_init = ser._open_zipfile_reader.__init__

    def reader_init(self, name_or_buffer):
        log("torch._C.PyTorchFileReader(f)   [C++ PyTorchStreamReader::init]")
        orig_reader_init(self, name_or_buffer)
        self.file_like = ReaderProxy(self.file_like)

    ser._open_zipfile_reader.__init__ = reader_init

    orig_load = ser._load

    def _load(zip_file, map_location, pickle_module, *args, **kwargs):
        log(f"_load(pickle_module={pickle_module.__name__})")
        global depth
        depth += 1
        try:
            return orig_load(zip_file, map_location, pickle_module, *args, **kwargs)
        finally:
            depth -= 1

    ser._load = _load

    orig_grl = ser._get_restore_location

    def _get_restore_location(map_location):
        log("_get_restore_location(map_location)")
        fn = orig_grl(map_location)
        return traced("restore_location", lambda storage, loc: f"location={loc!r}")(fn)

    ser._get_restore_location = _get_restore_location

    orig_rgi = wo._read_global_instruction

    def _read_global_instruction(readline):
        module, name = orig_rgi(readline)
        log(f"GLOBAL {module}.{name}")
        return module, name

    wo._read_global_instruction = _read_global_instruction

    orig_unpickler_load = wo.Unpickler.load

    def unpickler_load(self):
        self.persistent_load = traced(
            "persistent_load", lambda pid: f"{pid[0]!r}, {pid[1]}, key={pid[2]!r}, {pid[3]!r}, numel={pid[4]}"
        )(self.persistent_load)
        log("Unpickler.load()   [weights_only interpreter]")
        global depth
        depth += 1
        try:
            return orig_unpickler_load(self)
        finally:
            depth -= 1

    wo.Unpickler.load = unpickler_load

    torch._utils._rebuild_tensor_v2 = traced(
        "_rebuild_tensor_v2", lambda storage, off, size, stride, *a: f"size={tuple(size)}"
    )(torch._utils._rebuild_tensor_v2)
    torch._utils._validate_loaded_sparse_tensors = traced("_validate_loaded_sparse_tensors")(
        torch._utils._validate_loaded_sparse_tensors
    )

    nn.Module.load_state_dict = traced("Module.load_state_dict()")(nn.Module.load_state_dict)
    nn.Module._load_from_state_dict = traced(
        "_load_from_state_dict", lambda self, sd, prefix, *a: f"prefix={prefix!r}"
    )(nn.Module._load_from_state_dict)
    torch.Tensor.copy_ = traced("Tensor.copy_", lambda self, src, *a: f"shape={tuple(self.shape)}")(
        torch.Tensor.copy_
    )


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 3)


def collapse(lines):
    out = []
    for line in lines:
        if out and out[-1][0] == line:
            out[-1][1] += 1
        else:
            out.append([line, 1])
    return [f"{l}   (x{n})" if n > 1 else l for l, n in out]


def main():
    torch.manual_seed(0)
    path = Path(tempfile.mkdtemp()) / "tinynet_state_dict.pt"
    torch.save(TinyNet().state_dict(), path)

    install()

    print(f"torch {torch.__version__}: call order for torch.load('{path.name}')  (weights_only default)")
    print("=" * 78)
    log("torch.load(f)")
    global depth
    depth += 1
    sd = torch.load(path)
    depth -= 1
    log("model = TinyNet()")
    model = TinyNet()
    model.load_state_dict(sd)
    print("\n".join(collapse(events)))

    # GLOBAL look-ups are logged at the moment the pickle VM reaches them; the
    # allowlist dictionary lookup itself is a plain dict access (no import).
    print("=" * 78)
    print("GLOBAL opcodes seen, in order of first appearance:")
    seen = []
    for e in events:
        e = e.strip()
        if e.startswith("GLOBAL ") and e not in seen:
            seen.append(e)
    for e in seen:
        print("  " + e)


if __name__ == "__main__":
    main()
