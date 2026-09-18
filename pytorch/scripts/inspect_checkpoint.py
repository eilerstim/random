"""Empirically inspect what ``torch.save`` writes and what ``torch.load`` does.

Run:  python pytorch/scripts/inspect_checkpoint.py [out_dir]

It writes a few checkpoints of a tiny CNN into ``out_dir`` (default: a temp
dir) and prints, for each one:

* the ZIP member table (name, local-header offset, data offset, 64-byte
  alignment check, the ``FB`` padding extra field, STORED vs DEFLATED),
* the bookkeeping records (``version``, ``byteorder``, ``.format_version``,
  ``.storage_alignment``, ``.data/serialization_id``),
* a ``pickletools.dis`` disassembly of ``data.pkl`` (the interesting part),
* what the ``weights_only`` unpickler accepts and rejects.

Only the standard library and torch are needed.
"""

import io
import pickle
import pickletools
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
import torch.nn as nn

LOCAL_HEADER_SIZE = 30  # fixed part of a ZIP local file header


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3)
        self.bn = nn.BatchNorm2d(2)  # has parameters *and* buffers
        self.fc = nn.Linear(2, 3)

    def forward(self, x):
        x = self.bn(self.conv(x)).mean(dim=(2, 3))
        return self.fc(x)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def local_header(fh, header_offset):
    """Parse the 30-byte ZIP local file header the way PyTorchStreamReader::getRecordOffset does."""
    fh.seek(header_offset)
    hdr = fh.read(LOCAL_HEADER_SIZE)
    sig, _ver, _flags, method, _t, _d, _crc, _csize, _usize, fname_len, extra_len = struct.unpack("<4s5H3L2H", hdr)
    assert sig == b"PK\x03\x04", sig
    fname = fh.read(fname_len)
    extra = fh.read(extra_len)
    return method, fname, extra, header_offset + LOCAL_HEADER_SIZE + fname_len + extra_len


def dump_zip_table(path):
    with zipfile.ZipFile(path) as zf, open(path, "rb") as fh:
        print(f"{'member':<40} {'hdr_off':>8} {'data_off':>8} {'%64':>4} {'size':>8}  method  local extra field")
        for info in zf.infolist():
            method, _fname, extra, data_off = local_header(fh, info.header_offset)
            method_s = "STORED" if method == zipfile.ZIP_STORED else f"method={method}"
            extra_desc = ""
            if extra[:2] == b"FB":
                (pad_len,) = struct.unpack("<H", extra[2:4])
                extra_desc = f"'FB' + {pad_len} pad bytes ({len(extra)}B total)"
            elif extra:
                extra_desc = f"{len(extra)}B (id 0x{extra[1:2].hex()}{extra[0:1].hex()})"
            print(
                f"{info.filename:<40} {info.header_offset:>8} {data_off:>8} {data_off % 64:>4} "
                f"{info.file_size:>8}  {method_s:<7} {extra_desc}"
            )
        print()
        prefix = zf.namelist()[0].split("/")[0]
        for rec in ("version", "byteorder", ".format_version", ".storage_alignment", ".data/serialization_id"):
            name = f"{prefix}/{rec}"
            if name in zf.namelist():
                print(f"  {rec:<24} = {zf.read(name)!r}")
        return prefix


def dis_pickle(path, prefix, max_lines=None):
    with zipfile.ZipFile(path) as zf:
        data = zf.read(f"{prefix}/data.pkl")
    out = io.StringIO()
    pickletools.dis(data, out=out, annotate=0)
    lines = out.getvalue().splitlines()
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + [f"    ... ({len(lines) - max_lines} more lines)"]
    print("\n".join(lines))
    return data


def main(out_dir):
    torch.manual_seed(0)
    model = TinyNet()
    model(torch.randn(1, 1, 8, 8))  # bump BatchNorm's num_batches_tracked

    sd_path = out_dir / "tinynet_state_dict.pt"
    full_path = out_dir / "tinynet_full_model.pt"
    legacy_path = out_dir / "tinynet_legacy.pt"

    torch.save(model.state_dict(), sd_path)
    torch.save(model, full_path)
    torch.save(model.state_dict(), legacy_path, _use_new_zipfile_serialization=False)

    hr(f"torch {torch.__version__}: state_dict keys")
    for k, v in model.state_dict().items():
        print(f"  {k:<24} {str(tuple(v.shape)):<14} {v.dtype}")
    print(f"  _metadata = {dict(model.state_dict()._metadata)}")

    hr(f"ZIP layout of {sd_path.name} (torch.save(model.state_dict()))")
    print(f"first 4 bytes: {sd_path.read_bytes()[:4]!r}  (local file header magic checked by _is_zipfile)")
    prefix = dump_zip_table(sd_path)

    hr(f"pickletools.dis of {prefix}/data.pkl (state_dict)")
    dis_pickle(sd_path, prefix)

    hr(f"ZIP layout of {full_path.name} (torch.save(model))")
    prefix_full = dump_zip_table(full_path)

    hr(f"pickletools.dis of {prefix_full}/data.pkl (whole nn.Module) - first 60 lines")
    dis_pickle(full_path, prefix_full, max_lines=60)

    hr("GLOBAL opcodes referenced by each pickle")
    for p, pre in ((sd_path, prefix), (full_path, prefix_full)):
        with zipfile.ZipFile(p) as zf:
            data = zf.read(f"{pre}/data.pkl")
        globs = sorted({f"{arg}" for op, arg, _ in pickletools.genops(data) if op.name == "GLOBAL"})
        print(f"  {p.name}:")
        for g in globs:
            print(f"      {g}")

    hr("weights_only=True (default) behaviour")
    sd = torch.load(sd_path)  # default weights_only=True
    print(f"  state_dict loads fine: {type(sd).__name__} with {len(sd)} tensors")
    try:
        torch.load(full_path)
    except pickle.UnpicklingError as e:
        first = str(e).splitlines()
        print("  whole-module checkpoint is rejected:")
        for line in first[:4]:
            print("     ", line[:120])
    print(f"  get_unsafe_globals_in_checkpoint(full) = {torch.serialization.get_unsafe_globals_in_checkpoint(full_path)}")
    # Every class the pickle instantiates must be allowlisted, not just the top-level one.
    # ('__builtin__ set' is fine: IMPORT_MAPPING renames it to builtins.set, which is allowed.)
    with torch.serialization.safe_globals([TinyNet, nn.Conv2d, nn.BatchNorm2d, nn.Linear]):
        m = torch.load(full_path)
        print(f"  ...loads with safe_globals([TinyNet, Conv2d, BatchNorm2d, Linear]) => {type(m).__name__}")

    hr("Why weights_only exists: a pickle can call any importable callable")
    class Payload:
        def __reduce__(self):
            return (print, ("*** arbitrary code ran during torch.load(weights_only=False) ***",))

    buf = io.BytesIO()
    torch.save({"weights": torch.zeros(2), "oops": Payload()}, buf)
    buf.seek(0)
    try:
        torch.load(buf, weights_only=True)
    except pickle.UnpicklingError as e:
        print("  weights_only=True :", str(e).splitlines()[-1][:110])
    buf.seek(0)
    torch.load(buf, weights_only=False)

    hr("Buffer save uses archive name 'archive/'")
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        print("  members:", zf.namelist()[:3], "...")

    hr("Tensors that share a storage share one data/N record")
    base = torch.arange(10.0)
    view = base[2:5]
    buf = io.BytesIO()
    torch.save({"base": base, "view": view}, buf)
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        members = [n for n in zf.namelist() if "/data/" in n]
    print(f"  two tensors saved, storage records in the zip: {members}")
    buf.seek(0)
    loaded = torch.load(buf)
    same = loaded["view"].untyped_storage().data_ptr() == loaded["base"].untyped_storage().data_ptr()
    print(f"  after load they still share one storage: {same}; view.storage_offset() = {loaded['view'].storage_offset()}, "
          f"view.shape = {tuple(loaded['view'].shape)}")

    hr(f"Legacy (pre-1.6) format: {legacy_path.name}")
    raw = legacy_path.read_bytes()
    print(f"  first 4 bytes: {raw[:4]!r} (not a ZIP -> _legacy_load)")
    f = io.BytesIO(raw)
    magic = pickle.load(f)
    proto = pickle.load(f)
    sysinfo = pickle.load(f)
    print(f"  pickle #1 magic  = {hex(magic)} (== torch.serialization.MAGIC_NUMBER: {magic == torch.serialization.MAGIC_NUMBER})")
    print(f"  pickle #2 proto  = {proto}")
    print(f"  pickle #3 sysinfo= {sysinfo}")
    start = f.tell()
    pickletools.dis(f, out=io.StringIO())  # skip main object pickle (parses via genops)
    end = f.tell()
    print(f"  pickle #4 main object: bytes [{start}, {end})")
    keys = pickle.load(f)
    print(f"  pickle #5 storage keys (str(cdata) ids) = {keys[:3]}{' ...' if len(keys) > 3 else ''}")
    (n,) = struct.unpack("<q", f.read(8))
    print(f"  then raw storages: int64 numel={n} followed by {n} elements, repeated per key")

    hr("mmap=True maps the whole file once and slices it per storage")
    sd_mm = torch.load(sd_path, mmap=True)  # UntypedStorage.from_file(whole file, MAP_PRIVATE) + slicing
    w = sd_mm["conv.weight"]
    print(f"  conv.weight.untyped_storage().data_ptr() % 64 = {w.untyped_storage().data_ptr() % 64} "
          "(each data/N record starts on a 64-byte boundary, so the mapped pointer is aligned too)")
    print(f"  values identical to eager load: {torch.equal(w, sd['conv.weight'])}")

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="torch_ckpt_"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing checkpoints to {out}")
    main(out)
