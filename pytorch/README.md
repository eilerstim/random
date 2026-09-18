# PyTorch: how a model is structured and what `torch.load` really does

A source-level deep dive into PyTorch's native checkpoint format (`torch.save` / `torch.load`).

- **Version studied:** `torch 2.14.0+cpu` (wheel), source read at git tag `v2.14.0`. The Python files shipped in the wheel are byte-identical to the tag for every file cited here.
- **Citation format:** `path:line` refers to the PyTorch repository at `v2.14.0`. C++ files live under `caffe2/serialize/`, `torch/csrc/`, and `aten/src/ATen/`.
- **Evidence:** every claim about bytes on disk was checked with [`scripts/inspect_checkpoint.py`](scripts/inspect_checkpoint.py); its full output for this version is in [`scripts/inspect_checkpoint_output_torch-2.14.0.txt`](scripts/inspect_checkpoint_output_torch-2.14.0.txt).

## TL;DR

1. A `.pt` / `.pth` file written by `torch.save` is an **uncompressed ZIP64 archive** with a custom writer (`PyTorchStreamWriter`, built on miniz). Every entry is prefixed with an archive name (`model/...` for `model.pt`, `archive/...` when saving to a buffer). Entry data starts on **64-byte boundaries** so the file can be `mmap`ed.
2. The archive holds one **pickle** (`data.pkl`, protocol 2) that describes the saved Python object, plus **one raw byte blob per tensor storage** (`data/0`, `data/1`, ...). Tensors inside the pickle are not bytes; they are calls to `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...)` whose `storage` argument is a *persistent id* tuple `("storage", FloatStorage, "0", "cpu", numel)` pointing at `data/0`.
3. `torch.load` opens the zip in C++, reads `data.pkl` into memory, and runs an **unpickler** over it. Each persistent id triggers a read of `data/<key>` (or a slice of an `mmap`), `map_location` is applied to that storage, and `_rebuild_tensor_v2` wraps it into a tensor.
4. Since 2.6 the default unpickler is PyTorch's own **`weights_only` interpreter**: a re-implementation of the pickle VM that never imports anything, resolves `GLOBAL` opcodes through an allowlist, and only lets `REDUCE`/`NEWOBJ`/`BUILD` touch allowlisted types. With `weights_only=False` the standard `pickle.Unpickler` is used, and the file can execute arbitrary code.
5. Loading a checkpoint into a model is a second, separate step: `model.load_state_dict()` walks the module tree and `copy_()`s each tensor into the matching parameter or buffer, checking shapes and key names.

---

## Part 1: How a PyTorch model is structured

### 1.1 In memory: the `nn.Module` tree

An `nn.Module` is a plain Python object whose interesting state lives in a handful of dicts created in `Module.__init__` (`torch/nn/modules/module.py:482`):

| Attribute | Type | Purpose |
|---|---|---|
| `_parameters` | `dict[str, Parameter]` | learnable tensors (`Parameter` is a `Tensor` subclass, `torch/nn/parameter.py:30`) |
| `_buffers` | `dict[str, Tensor]` | non-learnable state such as BatchNorm running stats |
| `_non_persistent_buffers_set` | `set[str]` | buffers excluded from `state_dict()` |
| `_modules` | `dict[str, Module]` | child modules, giving the tree structure |
| `training` | `bool` | train/eval flag |
| `_forward_hooks`, `_forward_pre_hooks`, `_backward_hooks`, ... | `OrderedDict[int, Callable]` | hook registries |
| `_state_dict_hooks`, `_load_state_dict_pre_hooks`, ... | `OrderedDict[int, Callable]` | serialization hook registries |

`Module.__setattr__` (`module.py:1976`) is what makes `self.conv = nn.Conv2d(...)` register the child in `_modules` and `self.weight = nn.Parameter(...)` register in `_parameters`. Buffers are added explicitly with `register_buffer` (`module.py:528`).

A `Tensor` itself is metadata (`dtype`, `size`, `stride`, `storage_offset`, `device`, `requires_grad`) on top of an `UntypedStorage`, which is the actual byte buffer. Several tensors can share one storage (views). `TypedStorage` (`torch/storage.py:685`) is a deprecated Python wrapper that still matters here because the on-disk format speaks in typed storage names such as `FloatStorage`.

### 1.2 The `state_dict`: the recommended thing to save

`Module.state_dict()` (`module.py:2199`) flattens the tree into a single `OrderedDict`:

- Keys are dotted paths (`conv.weight`, `bn.running_mean`), built by recursing over `_modules` with a growing `prefix` (`module.py:2272`).
- `_save_to_state_dict` (`module.py:2148`) adds every parameter and every *persistent* buffer, detached from autograd (`param.detach()`), plus an optional `<prefix>_extra_state` entry if the module overrides `get_extra_state`.
- The dict carries an extra attribute `_metadata`: `{prefix: {"version": module._version}}` for every module (`module.py:2265`). Modules bump `_version` when their state layout changes; e.g. `BatchNorm` has `_version = 2` (`torch/nn/modules/batchnorm.py:28`) and its `_load_from_state_dict` synthesizes a missing `num_batches_tracked` when it sees a version `< 2` (`batchnorm.py:129`).

For the tiny CNN in the experiment script this looks like:

```
conv.weight              (2, 1, 3, 3)   torch.float32
conv.bias                (2,)           torch.float32
bn.weight                (2,)           torch.float32
bn.bias                  (2,)           torch.float32
bn.running_mean          (2,)           torch.float32
bn.running_var           (2,)           torch.float32
bn.num_batches_tracked   ()             torch.int64
fc.weight                (3, 2)         torch.float32
fc.bias                  (3,)           torch.float32
_metadata = {'': {'version': 1}, 'conv': {'version': 1}, 'bn': {'version': 2}, 'fc': {'version': 1}}
```

The state dict contains **no architecture**. Rebuilding the model requires the Python class that produced it. That is also why it is the safe thing to serialize: it is only dicts, strings and tensors.

### 1.3 Saving a whole `nn.Module` instead

`torch.save(model)` pickles the module object itself. Because `Module` defines `__getstate__`/`__setstate__` (`module.py:1924`, `module.py:1929`) but no `__reduce__`, the default protocol-2 reduction applies: the pickle stores a **reference to the class by module path and name** (`GLOBAL __main__ TinyNet`), then `NEWOBJ` (calls `TinyNet.__new__`), then the whole `__dict__` (every attribute from the table above, including nested `Conv2d`/`BatchNorm2d`/`Linear` objects and the hook dicts), then `BUILD`, which calls `Module.__setstate__` to install it. `__setstate__` also back-fills attributes that older PyTorch versions did not have.

Consequences:

- Loading needs the exact class importable under the same module path (`__main__.TinyNet` here, which is why models saved from scripts are hard to load elsewhere).
- Parameters are pickled through `Parameter.__reduce_ex__` (`parameter.py:88`) as `torch._utils._rebuild_parameter(data, requires_grad, OrderedDict())`, where `data` is a regular tensor.
- Every class in the file must be allowlisted for `weights_only=True`. The experiment shows `get_unsafe_globals_in_checkpoint` reporting `TinyNet`, `Conv2d`, `BatchNorm2d` and `Linear`, and the load only succeeding inside `safe_globals([TinyNet, nn.Conv2d, nn.BatchNorm2d, nn.Linear])`.

### 1.4 How a tensor becomes pickle opcodes

`torch.save` uses the standard pickle machinery with two hooks.

**Hook 1: `Tensor.__reduce_ex__`** (`torch/_tensor.py:266` → `_reduce_ex_internal`, `torch/_tensor.py:315`). For an ordinary dense tensor it returns

```python
(torch._utils._rebuild_tensor_v2,
 (TypedStorage(wrap_storage=<untyped storage>, dtype=self.dtype),
  self.storage_offset(), tuple(self.size()), self.stride(),
  self.requires_grad, OrderedDict()))            # empty backward hooks, never serialized (see Note [Don't serialize hooks], _utils.py)
```

so what ends up in the pickle is a *call* to `_rebuild_tensor_v2` whose first argument is a storage object. Tensor subclasses or tensors with Python attributes are wrapped one level further in `torch._tensor._rebuild_from_type_v2(func, type, args, state)` (`_tensor.py:56`).

**Hook 2: `persistent_id`** (`torch/serialization.py:1199`). `torch.save` installs a `Pickler.persistent_id` that intercepts every storage object before it would be pickled normally and returns the tuple

```python
("storage", torch.FloatStorage, "0", "cpu", 18)
#            ^ legacy typed-storage class  ^ key  ^ location tag  ^ numel (elements, not bytes)
```

The pickler emits this tuple followed by a `BINPERSID` opcode instead of the storage bytes. On the side, `_save` remembers `serialized_storages["0"] = <untyped storage>` and later writes its bytes as the zip entry `data/0`.

Details that matter:

- The key is assigned by `id_map.setdefault(storage._cdata, str(len(id_map)))` (`serialization.py:1234`): storages are numbered in order of first appearance, and a storage shared by several tensors (views) is written **once**. Views are restored via their `storage_offset`/`size`/`stride` arguments. This also means saving a small slice of a huge tensor writes the whole underlying storage (`docs/source/notes/serialization.md`, "Saving and loading tensors preserves views").
- The location tag comes from `location_tag()` (`serialization.py:701`), which asks the registered taggers in priority order (`cpu` first, then `cuda`, `mps`, `meta`, `privateuse1`, `hpu`, `xpu`, `mtia`; `serialization.py:671-699`). CUDA storages get tags like `cuda:0`.
- The typed storage class name is a legacy artifact: `TypedStorage._pickle_storage_type()` maps `dtype → "FloatStorage"` through `_dtype_to_storage_type_map()` (`storage.py:572`). Newer dtypes (`float8_*`, `uint16/32/64`, `bits*`, ...) are not in that map; for them `_reduce_ex_internal` uses `_rebuild_tensor_v3` with an `UntypedStorage` plus an explicit `dtype` argument (`_tensor.py:497-499`, `_utils.py:253`).
- `torch.save` pickles with protocol 2 (`DEFAULT_PROTOCOL = 2`, `serialization.py:59`). A side effect is Python-2-era names such as `GLOBAL __builtin__ set` in module pickles; the loader maps them back through `IMPORT_MAPPING`/`NAME_MAPPING` (`_utils.py:1121`, `_utils.py:1140`).

Other tensor kinds take other rebuild functions, all in `torch/_utils.py`:

| Tensor kind | Reduced to | Notes |
|---|---|---|
| dense, legacy dtype | `_rebuild_tensor_v2` (`_utils.py:230`) | the common case |
| dense, new dtype | `_rebuild_tensor_v3` (`_utils.py:253`) | `UntypedStorage` + `dtype` argument |
| `Parameter` | `_rebuild_parameter` / `_rebuild_parameter_with_state` (`_utils.py:518`, `:528`) | wraps a rebuilt tensor |
| sparse COO / CSR / CSC / BSR / BSC | `_rebuild_sparse_tensor` (`_utils.py:354`) | built with `check_invariants=False`, validated in bulk afterwards |
| quantized | `_rebuild_qtensor` (`_utils.py:449`) | carries `qscheme`, scales, zero points |
| meta device | `_rebuild_meta_tensor_no_storage` (`_utils.py:418`) | no storage at all |
| nested | `_rebuild_nested_tensor` (`_utils.py:396`) | |
| wrapper subclasses (`__torch_dispatch__`) | `_rebuild_wrapper_subclass` (`_utils.py:424`) | |
| XLA / MAIA / MTIA, no storage | `_rebuild_device_tensor_from_cpu_tensor` (`_utils.py:400`) | saved as a CPU tensor |

### 1.5 On disk: the zip container

![PyTorch checkpoint structure](figures/pytorch_structure.svg?v=acd7ae7)

`torch.save` (`serialization.py:944`) opens a `torch._C.PyTorchFileWriter` (`serialization.py:810`, binding at `torch/csrc/jit/python/init.cpp:1402`) whose C++ implementation is `caffe2::serialize::PyTorchStreamWriter` (`caffe2/serialize/inline_container.cc:693`). Then `_save` (`serialization.py:1183`) writes the records in this order:

| Zip entry (`<archive>/...`) | Written by | Content | Since |
|---|---|---|---|
| `data.pkl` | `_save`, `serialization.py:1259` | the pickle described above, without storage bytes | 1.6 |
| `.format_version` | `_save`, `serialization.py:1265` | `"1"`: storages are stored in numeric key order, which lets the loader compute storage offsets without random reads | 2.7 |
| `.storage_alignment` | `_save`, `serialization.py:1268` | `"64"` (configurable via `torch.utils.serialization.config.save.storage_alignment`) | 2.7 |
| `byteorder` | `_save`, `serialization.py:1276` | `"little"` or `"big"` (`sys.byteorder`) | 2.1 |
| `data/<key>` | `_save`, `serialization.py:1312` | raw storage bytes, copied to CPU first if needed; one entry per storage | 1.6 |
| `version` | `PyTorchStreamWriter::writeEndOfFile`, `inline_container.cc:855` | `"3\n"` (`kMinProducedFileFormatVersion = 3`, `caffe2/serialize/versions.h:81`); TorchScript archives may carry up to 10 | 1.6 |
| `.data/serialization_id` | `writeSerializationId`, `inline_container.cc:904` | 40 decimal digits: a combined hash of all record names (20 digits) followed by a combined CRC32 of all uncompressed record data (20 digits) | 2.1 |

Observed layout of `tinynet_state_dict.pt` (from the script; `data_off` is the byte offset of the entry's data):

```
member                                    hdr_off data_off  %64     size  method  local extra field
tinynet_state_dict/data.pkl                     0       64    0      854  STORED  'FB' + 3 pad bytes (7B total)
tinynet_state_dict/.format_version            934     1024    0        1  STORED  'FB' + 22 pad bytes (26B total)
tinynet_state_dict/.storage_alignment        1041     1152    0        2  STORED  'FB' + 40 pad bytes (44B total)
tinynet_state_dict/byteorder                 1170     1280    0        6  STORED  'FB' + 48 pad bytes (52B total)
tinynet_state_dict/data/0                    1302     1408    0       72  STORED  'FB' + 47 pad bytes (51B total)
tinynet_state_dict/data/1                    1496     1600    0        8  STORED  'FB' + 45 pad bytes (49B total)
...
tinynet_state_dict/data/8                    2408     2496    0       12  STORED  'FB' + 29 pad bytes (33B total)
tinynet_state_dict/version                   2524     2624    0        2  STORED  'FB' + 40 pad bytes (44B total)
tinynet_state_dict/.data/serialization_id     2642     2752    0       40  STORED  'FB' + 35 pad bytes (39B total)

version                  = b'3\n'
byteorder                = b'little'
.format_version          = b'1'
.storage_alignment       = b'64'
.data/serialization_id   = b'0197786035336920144316088823782960924604'
```

Properties of the container (`inline_container.h` header comment, and the code):

- **Archive name prefix.** `PyTorchStreamWriter(file_name)` uses `basename()` without extension as the folder (`inline_container.cc:74`): `model.pt` → `model/data.pkl`. When writing to a Python buffer the name is the literal `archive` (`inline_container.cc:703`). The reader recovers the prefix from the first entry (`inline_container.cc:182`) and prepends it to every lookup, so the prefix can be anything.
- **64-byte alignment.** `writeRecord` (`inline_container.cc:773`) computes how many bytes the entry's data start would be off a 64-byte boundary and stuffs that many filler bytes into the local header's *extra field*, tagged with the two-byte id `FB` (`detail::getPadding`, `inline_container.cc:284`). This is a standard, tool-compatible zip trick; `zipfile.ZipFile` skips the extra field and reads the entries fine.
- **No compression.** Records are added with compression level 0 unless `compress=True`, which `torch.save` never passes (`inline_container.cc:809`). A CRC32 is still computed unless `torch.utils.serialization.config.save.compute_crc32` is off.
- **Always ZIP64** (`MZ_ZIP_FLAG_WRITE_ZIP64`, `inline_container.cc:765`), so checkpoints larger than 4 GiB are fine.
- **Single pass.** Zip puts its central directory at the end, so the writer never seeks back. Metadata records (`version`, `byteorder` fallback, serialization id) are appended in `writeEndOfFile` (`inline_container.cc:828`), then `mz_zip_writer_finalize_archive` writes the central directory.

### 1.6 Inside `data.pkl`

`pickletools.dis` of the state-dict pickle above, first tensor only:

```
    0: \x80 PROTO      2
    2: c    GLOBAL     'collections OrderedDict'
   29: )    EMPTY_TUPLE
   30: R    REDUCE                               -> OrderedDict()
   33: (    MARK
   34: X        BINUNICODE 'conv.weight'
   52: c        GLOBAL     'torch._utils _rebuild_tensor_v2'
   87: (        MARK
   88: (            MARK
   89: X                BINUNICODE 'storage'
  103: c                GLOBAL     'torch FloatStorage'
  125: X                BINUNICODE '0'          <- key: zip entry data/0
  133: X                BINUNICODE 'cpu'        <- location tag
  143: K                BININT1    18           <- numel (18 floats = 72 bytes)
  145: t                TUPLE      (MARK at 88)
  148: Q            BINPERSID                    -> persistent_load(("storage", FloatStorage, "0", "cpu", 18))
  149: K            BININT1    0                <- storage_offset
  151: (            MARK
  152: K                BININT1    2
  154: K                BININT1    1
  156: K                BININT1    3
  158: K                BININT1    3
  160: t                TUPLE      (MARK at 151) <- size (2, 1, 3, 3)
  163: (            MARK  ... 9 9 3 1 ... t      <- stride (9, 9, 3, 1)
  175: \x89         NEWFALSE                     <- requires_grad (state_dict tensors are detached)
  176: h            BINGET     0
  178: )            EMPTY_TUPLE
  179: R            REDUCE                       -> OrderedDict()  (empty backward_hooks)
  182: t            TUPLE      (MARK at 87)
  185: R        REDUCE                           -> _rebuild_tensor_v2(...)
  ...
  743: u        SETITEMS   (MARK at 33)          -> fills the OrderedDict
  744: }    EMPTY_DICT ... '_metadata' ...
  852: b    BUILD                                -> OrderedDict.__dict__.update({'_metadata': {...}})
  853: .    STOP
```

The complete set of `GLOBAL`s in a plain state dict is tiny: `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, and one legacy storage class per dtype (`torch.FloatStorage`, `torch.LongStorage`). The whole-module pickle adds `__main__.TinyNet`, `torch.nn.modules.conv.Conv2d`, `torch.nn.modules.batchnorm.BatchNorm2d`, `torch.nn.modules.linear.Linear`, `torch._utils._rebuild_parameter` and `__builtin__.set`.

### 1.7 The legacy (pre-1.6) formats

`torch.save(..., _use_new_zipfile_serialization=False)` still writes the old format (`_legacy_save`, `serialization.py:1021`), and `torch.load` still reads it. It is a sequence of pickles followed by raw bytes in one flat file:

```
pickle #1  MAGIC_NUMBER            0x1950a86a20f9469cfc6c        (serialization.py:65)
pickle #2  PROTOCOL_VERSION        1001                          (serialization.py:66)
pickle #3  sys_info                {'protocol_version': 1001, 'little_endian': True, 'type_sizes': {...}}
pickle #4  the object              same persistent-id scheme, key = str(storage._cdata), plus view metadata
pickle #5  sorted storage keys     e.g. ['188986384', '188986960', ...]
raw        for each key: int64 numel, then numel * element_size bytes   (THPStorage_writeFileRaw, torch/csrc/serialization.cpp:235)
```

There is an even older tar-based layout (`storages`, `tensors`, `pickle` members) handled by `legacy_load` inside `_legacy_load` (`serialization.py:1740`). Neither legacy layout supports `mmap=True`, and the tar layout refuses `weights_only=True` outright (`serialization.py:1759`).

### 1.8 Other files `torch.load` accepts

- **TorchScript archives** (`torch.jit.save`) are zips too, with `constants.pkl`, `code/*.py` and `data.pkl`. `torch.load` detects them by the presence of `constants.pkl` (`_is_torchscript_zip`, `serialization.py:2251`), warns, and tail-calls `torch.jit.load`. With `weights_only=True` it raises instead (`serialization.py:1583`).
- **safetensors**: since 2.13, a path ending in `.safetensors` is forwarded to `safetensors.torch.load_file` (`serialization.py:1536`). Only string/device `map_location` is supported on that path.

---

## Part 2: What happens inside `torch.load`

### 2.1 The flow

```
model.pt
│
├── data.pkl
├── data/0 … data/N
├── byteorder
└── version
        │
        ▼
① torch.load()
        │
        ▼
② Open the .pt ZIP
        │
        ▼
③ Read data.pkl
        │
        ▼
④ Pick the unpickler
        │
        ├── weights_only=True  → allowlist unpickler
        └── weights_only=False → pickle.Unpickler
        │
        ▼
⑤ unpickler.load()
        │
        ├── Look up collections.OrderedDict
        │
        ├── Look up torch._utils._rebuild_tensor_v2
        │
        └── Look up torch.FloatStorage
        │
        ▼
⑥ persistent_load()
        │
        ├── read data/N
        └── apply map_location
        │
        ▼
Create each Storage
        │
        ▼
⑦ _rebuild_tensor_v2()
        │
        ▼
Create each Tensor
e.g.:
conv.weight
conv.bias
bn.running_mean
...
        │
        ▼
Complete state_dict
        │
        ▼
⑧ model = MyNet()
        │
        ▼
⑨ model.load_state_dict()
        │
        ├── match keys
        ├── check shapes
        └── param.copy_()
        │
        ▼
Final model
```

### 2.2 Step by step, with the code

**① `torch.load()`** (`serialization.py:1315`). If `weights_only` is not passed, `_default_to_weights_only(pickle_module)` (`serialization.py:89`) returns `True` unless a custom `pickle_module` was given (then it silently becomes `False`). Two environment variables can override: `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` forces `True` everywhere; `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` forces `False` only where the caller did not pass the argument (`serialization.py:1483-1508`). Passing `pickle_module` together with `weights_only=True` is an error. `encoding="utf-8"` is added to the unpickler arguments by default. If `f` is a path ending in `.safetensors`, the function returns early via safetensors (`serialization.py:1536`).

The file is then opened with `_open_file_like` (`serialization.py:793`): a path with `open(name, "rb")`, a file-like object after checking that it supports `seek`/`tell` (`_check_seekable`), because the zip reader needs random access. `_is_zipfile` (`serialization.py:428`) compares the first four bytes with `PK\x03\x04`. Anything else goes to `_legacy_load` (`serialization.py:1667`), which first tries the tar layout and then the magic-number pickle sequence.

**② Open the .pt ZIP** (`_open_zipfile_reader`, `serialization.py:805` → `torch._C.PyTorchFileReader`, `init.cpp:1593`). For a path the C++ side uses a `FileAdapter` (`fopen`/`fseeko`/`fread`, `caffe2/serialize/file_adapter.cc`); for a Python buffer, a `BufferAdapter` that calls the object's `seek`/`readinto`/`read` (`init.cpp:1530`). `PyTorchStreamReader::init` (`inline_container.cc:142`):

1. rejects files starting with the 2018 preview magic `PYTORCH1` (`inline_container.cc:157`);
2. `mz_zip_reader_init` parses the central directory (`inline_container.cc:164`);
3. takes the first entry's name up to the first `/` as `archive_name_` (`inline_container.cc:182`). A file whose first entry has no `/` is rejected;
4. reads `.data/serialization_id` if present (`inline_container.cc:186`) and logs it;
5. reads `.data/version` or `version` (`inline_container.cc:204`), parses it with `stoull`, and requires `1 <= version <= 10` (`kMinSupportedFileFormatVersion`, `kMaxSupportedFileFormatVersion`, `versions.h:7-9`).

Record access goes through `getRecord(name)` (`inline_container.cc:370`): locate the entry, allocate `m_uncomp_size` bytes with the CPU allocator, and `mz_zip_reader_extract_to_mem` into it. The reader can therefore also read compressed entries produced by other tools, even though PyTorch never writes them.

Two things happen right after the archive is open. `_is_torchscript_zip` (`serialization.py:2251`) checks for `constants.pkl`; if present, `torch.load` rewinds and hands over to `torch.jit.load` (or raises under `weights_only=True`, `serialization.py:1583`). And if `mmap=True` (or `config.load.mmap`), `torch.load` requires a real path and maps the **whole file** once with `torch.UntypedStorage.from_file(path, shared, size)` (`serialization.py:1598` → `THPStorage_fromFile`, `torch/csrc/StorageMethods.cpp:410` → `at::MapAllocator`, `mmap(nullptr, size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0)`, `aten/src/ATen/MapAllocator.cpp:351`). `MAP_SHARED` can be selected with `torch.serialization.set_default_mmap_options`.

**③ Read data.pkl** (`_load`, `serialization.py:1994`). `_load` first reads the bookkeeping records and prepares the storage loader:

- `.format_version` decides whether storage offsets may be *computed* instead of read (`serialization.py:2013`).
- The `byteorder` record is read (`serialization.py:2019`). If absent, the fallback comes from `torch.utils.serialization.config.load.endianness` (default: assume little endian).
- `restore_location = _get_restore_location(map_location)` (`serialization.py:1952`) turns `map_location` into a function `(storage, location_tag) -> storage`: `None` uses the registry defaults; a `dict` remaps tags; a string or `torch.device` sends everything to that device; a callable is tried first and falls back to the default when it returns `None`.

Then `data.pkl` is read fully into a `BytesIO` (`serialization.py:2233`). The pickle is never streamed.

**④ Pick the unpickler** (`serialization.py:2219`). Both paths construct a small `UnpicklerWrapper` subclass of `pickle_module.Unpickler`. With `weights_only=False` that is the standard library unpickler (or `dill`'s, etc.), and the wrapper's `find_class` returns a `StorageType(name)` (`serialization.py:1982`) for any global whose name contains `Storage`, so that `torch.FloatStorage` resolves to a lightweight object with a `.dtype` instead of the deprecated storage class; it also maps the old module name `torch.tensor` to `torch._tensor`. With `weights_only=True`, `pickle_module` is `torch._weights_only_unpickler` (described in 2.3); its interpreter never calls `find_class`, because its allowlist already maps the storage class names to `StorageType` objects. `torch.load`'s `persistent_load` is attached to the unpickler either way (`serialization.py:2236`).

**⑤ `unpickler.load()`** (`serialization.py:2241`). The unpickler executes the opcodes of `data.pkl` in order. `GLOBAL module name` pushes a callable or class: the standard unpickler imports the module (executing its top-level code) and fetches the attribute, while the `weights_only` unpickler looks the dotted name up in a dictionary and never imports (`_weights_only_unpickler.py:331`). For a state dict the look-ups are only `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` and one storage class per dtype. Two opcodes do the real work and get their own steps below: `BINPERSID` (⑥) and `REDUCE` (⑦). `SETITEMS` fills the `OrderedDict`; `BUILD` on the `OrderedDict` installs `_metadata` (the `weights_only` unpickler special-cases this as `inst.__dict__.update(state)`, `_weights_only_unpickler.py:428`). For whole modules, `NEWOBJ` creates the instance via `cls.__new__` and `BUILD` calls `Module.__setstate__` with the pickled `__dict__`.

When `STOP` is reached, `_load` runs `torch._utils._validate_loaded_sparse_tensors()` (`serialization.py:2244`), which checks the invariants of any sparse tensors that were built with `check_invariants=False` (always under `weights_only=True`), logs the serialization id, and returns the object. `torch.load` returns whatever was pickled: an `OrderedDict`, a module, a dict with `epoch`/`optimizer` keys, or any other Python object.

**⑥ `persistent_load()`** (`serialization.py:2183`). `BINPERSID` pops the tuple and calls it. It asserts the tuple starts with `"storage"`, extracts `(storage_type, key, location, numel)`, computes `nbytes = numel * element_size(dtype)`, and calls `load_tensor` (`serialization.py:2113`) unless the key was already loaded (`loaded_storages` cache, so shared storages are read once). `load_tensor`:

- normal path: `zip_file.get_storage_from_record("data/<key>", nbytes, torch.UntypedStorage)` (`serialization.py:2148`; binding at `init.cpp:1611`) reads the record into a fresh CPU storage and checks that the record size equals `nbytes`;
- `mmap` path: `overall_storage[offset : offset + nbytes]`, where `offset` is either read from the entry's local header (`getRecordOffset`, `inline_container.cc:622`) or computed arithmetically from the previous storage's offset when `config.load.calculate_storage_offsets` is on (`_get_offset`, `serialization.py:2066`, which mirrors miniz's header layout);
- meta/fake-tensor paths allocate an empty `meta` storage and only record the checkpoint offset;
- byteswaps in place if the file's byte order differs from the host (`storage.byteswap(dtype)`, `serialization.py:2155` → `THPStorage_byteswap`, `StorageMethods.cpp:618`);
- applies `restore_location(storage, location)` (`serialization.py:2167`), which is where CUDA tensors are moved to the GPU or remapped by `map_location`, and where a missing device raises the well-known "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False" error (`_validate_device`, `serialization.py:601`);
- wraps the result in a `TypedStorage` so that `_rebuild_tensor_v2` can read `.dtype` from it.

The result of this step is one storage object per `data/N` entry, already on its final device.

**⑦ `_rebuild_tensor_v2()`** (`_utils.py:230`). `REDUCE` calls the function on the stack with `(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None)`. It creates `torch.empty((0,), dtype, device=storage.device)`, calls `set_(storage, offset, size, stride)`, and sets `requires_grad`. No bytes are copied here; the tensor is a view on the storage created in ⑥, which is also how several tensors end up sharing one storage. Once `SETITEMS` has placed every tensor under its key, the `OrderedDict` is the complete state dict.

**⑧ `model = MyNet()`**. Nothing in the file describes the architecture, so the caller constructs the model from code. Its parameters hold freshly initialised values at this point; `torch.load` has not touched the model.

**⑨ `model.load_state_dict()`** (`Module.load_state_dict`, `module.py:2535`). `torch.load` does not know about your model; `load_state_dict` does the matching:

1. The input dict is shallow-copied into an `OrderedDict` and `_metadata` is carried over.
2. A recursive `load(module, local_state_dict, prefix)` (`module.py:2589`) visits every module; each child gets the subset of keys starting with its prefix.
3. `_load_from_state_dict` (`module.py:2350`) per module: runs `_load_state_dict_pre_hooks` (this is where `BatchNorm` patches old checkpoints using `local_metadata["version"]`), then for every local parameter and persistent buffer looks up `prefix + name`, checks it is tensor-like, checks the shape (`size mismatch for ...` errors), and copies with `param.copy_(input_param)` under `torch.no_grad()` (`module.py:2500`). With `assign=True` the checkpoint tensor is installed with `setattr` instead of being copied (keeping its dtype/device, and turning it into a `Parameter` if needed); with `torch.__future__.set_swap_module_params_on_conversion(True)` the tensors are swapped via `param.module_load`. `set_extra_state` is called if the module defines it.
4. `strict=True` collects `missing_keys` and `unexpected_keys` and raises one `RuntimeError` listing all problems (`module.py:2644`); with `strict=False` they are returned in an `_IncompatibleKeys` named tuple.

Because the copy happens **into the existing parameters**, the model must already be constructed with matching shapes, and dtype/device are those of the model, not of the checkpoint (unless `assign=True`).

### 2.3 The `weights_only` unpickler in detail

`torch/_weights_only_unpickler.py` is a from-scratch pickle VM (`class Unpickler`, line 307; `load`, line 315) that only understands the opcodes protocol 2 needs for tensors and containers: `PROTO`, `STOP`, `GLOBAL`, `NEWOBJ`, `REDUCE`, `BUILD`, `MARK`, `TUPLE*`, `EMPTY_*`, `APPEND(S)`, `SETITEM(S)`, `BINPERSID`, the integer/float/string opcodes, memo `BINPUT`/`BINGET`, and `LONG1`. Any other opcode raises `UnpicklingError("Unsupported operand ...")` (line 563). `STACK_GLOBAL`, `INST`, `OBJ`, `EXT*`, `BINBYTES`, `NEWOBJ_EX` and friends are simply not implemented. It warns if the pickle protocol is not 2 (line 549).

Its rules:

| Opcode | Rule (`_weights_only_unpickler.py`) |
|---|---|
| `GLOBAL` (line 331) | The dotted name is looked up in `_get_allowed_globals()` (line 174) or the user allowlist. Names from `sys`, `os`, `posix`, `nt` are rejected even if allowlisted (`_blocklisted_modules`, line 80). Nothing is imported; a miss raises with an actionable message naming the global. |
| `NEWOBJ` (line 384) | Only `torch.nn.Parameter` or allowlisted classes; `cls.__new__(cls, *args)` is called. |
| `REDUCE` (line 402) | The callable must be allowlisted; it is then called with the popped args. |
| `BUILD` (line 419) | `torch.Tensor` (legacy `set_`), `Parameter.__setstate__`, `OrderedDict.__dict__.update`, or an allowlisted type (its `__setstate__`, or a dict/slots update mimicking `pickle.load_build`). |
| `APPEND(S)` (line 579) | Only onto plain lists or allowlisted list subclasses. |
| `SETITEM(S)` (line 572) | Only into `dict`, `OrderedDict`, `Counter`. |
| `BINPERSID` (line 520) | The id must be a tuple whose first element is `"storage"` (or an int for legacy files); then `torch.load`'s `persistent_load` runs. |

The default allowlist (`_get_allowed_globals`) contains: `collections.OrderedDict`, `collections.Counter`, `torch.nn.parameter.Parameter`, `torch.Tensor` and all `torch._tensor_classes`, all storage classes (wrapped as `StorageType`), every `torch.dtype`, `torch.Size`, `torch.device`, the quantization schemes, `torch.serialization._get_layout`, `_codecs.encode`, `builtins.bytearray/set/complex`, the rebuild functions listed in `_tensor_rebuild_functions()` (line 150), and `torch._tensor._rebuild_from_type_v2`. Users extend it with `torch.serialization.add_safe_globals([...])` or the `safe_globals([...])` context manager, and can inspect a file without executing it via `torch.serialization.get_unsafe_globals_in_checkpoint(f)`, which statically scans the opcode stream (`get_globals_in_pkl`, line 242).

What it does not do, per the official note (`docs/source/notes/serialization.md`, "weights_only security"): it does not protect against denial of service (a pickle can still ask for a 100 GB storage), and memory corruption "might still be possible". PyTorch has been hardening the rebuild functions accordingly, e.g. `_rebuild_qtensor` now bounds-checks `axis` and the scales/zero-points lengths (`_utils.py:473-489`), and sparse tensors are always validated after a `weights_only` load.

### 2.4 `map_location` and the device registry

`torch.serialization.register_package(priority, tagger, deserializer)` (`serialization.py:444`) maintains a sorted list `_package_registry`. At save time `location_tag(storage)` returns the first non-`None` tagger result; at load time `default_restore_location(storage, location)` (`serialization.py:713`) returns the first non-`None` deserializer result. Built-ins (priority in parentheses): `cpu` (10), `cuda` (20), `mps` (21), `meta` (22), `privateuse1` (23), `hpu` (24), `xpu` (25), `mtia` (26). The CUDA deserializer validates that the device module exists, is available, and has enough devices before calling `storage.to(device)` (`_validate_device`, `serialization.py:601`). Out-of-tree backends register their own pair. `map_location` is applied per storage, right after the bytes are read and before any tensor is built.

---

## Part 3: Security-relevant observations

These follow directly from the mechanics above; they are the reason the `GLOBAL/REDUCE` box is red in the structure figure.

1. **`data.pkl` is executable.** With `weights_only=False`, `GLOBAL` imports any module (running its top-level code) and `REDUCE` calls any callable with attacker-chosen arguments. The experiment script demonstrates this with a harmless `print` payload embedded next to real weights: the same file is rejected by `weights_only=True` and executes under `weights_only=False`. A whole-module checkpoint necessarily exercises this path, because it must import the model's classes.
2. **`weights_only=True` is the mitigation, and the default since 2.6.** It is an allowlist interpreter, not a sandbox: it never imports, and it only calls allowlisted callables. Its guarantees stop at code execution; resource exhaustion and malformed tensor metadata are separate concerns that PyTorch addresses piecemeal (size checks in `get_storage_from_record`, sparse invariant validation, quantized-tensor bounds checks).
3. **Ways the mitigation is switched off.** Explicit `weights_only=False`; passing any `pickle_module` (which makes the default `False`); the environment variable `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; allowlisting a class whose `__setstate__`, `__new__` or `__reduce__` result does something dangerous; loading a TorchScript archive (`torch.jit.load` has its own C++ unpickler and no `weights_only`); and loading the legacy tar format, which requires `weights_only=False`.
4. **The container is parsed in C++ before any Python runs.** `PyTorchStreamReader::init` trusts the zip central directory (miniz), the first entry's name, the `version` string, and record sizes. `get_storage_from_record` does check that the entry size matches `numel * element_size`.
5. **Static inspection is possible without executing anything.** `zipfile` lists the entries; `pickletools.dis` disassembles `data.pkl`; `torch.serialization.get_unsafe_globals_in_checkpoint` names every global outside the allowlist. The experiment script does all three.
6. **Practical guidance that falls out of the format:** save and share `state_dict`s, not modules; load with `weights_only=True` (the default) and `map_location="cpu"`; prefer safetensors when interoperability matters, since it has no code path at all.

---

## Part 4: Reproduce it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # any 2.x works; 2.14.0 was used here
python pytorch/scripts/inspect_checkpoint.py            # writes checkpoints to a temp dir and dumps everything
python tools/structure_figure.py pytorch/figures/pytorch_structure.json -o pytorch/figures/pytorch_structure.svg
node tools/render_png.cjs pytorch/figures/pytorch_structure.svg pytorch/figures/pytorch_structure.png   # optional PNG (needs playwright)
```

The script prints the zip table, the record contents, the `pickletools` disassembly, the `weights_only` rejection and allowlisting behaviour, the harmless code-execution demo, the legacy header, and the `mmap` alignment check.

## Part 5: Source map (tag `v2.14.0`)

| What | Where |
|---|---|
| `torch.save`, `_save`, `persistent_id` | `torch/serialization.py:944`, `:1183`, `:1199` |
| `torch.load`, `weights_only` resolution, safetensors dispatch | `torch/serialization.py:1315`, `:1481-1508`, `:1536` |
| `_is_zipfile`, `_is_torchscript_zip` | `torch/serialization.py:428`, `:2251` |
| `_load`, `load_tensor`, `persistent_load`, `UnpicklerWrapper.find_class` | `torch/serialization.py:1994`, `:2113`, `:2183`, `:2223` |
| `_get_restore_location`, `register_package`, built-in taggers | `torch/serialization.py:1952`, `:444`, `:671-699` |
| `_legacy_save`, `_legacy_load`, `MAGIC_NUMBER` | `torch/serialization.py:1021`, `:1667`, `:65` |
| `weights_only` unpickler, allowlist, static scanner | `torch/_weights_only_unpickler.py:307`, `:174`, `:242` |
| `Tensor.__reduce_ex__`, `_reduce_ex_internal`, `_rebuild_from_type_v2` | `torch/_tensor.py:266`, `:315`, `:56` |
| `_rebuild_tensor_v2/v3`, `_rebuild_parameter`, other rebuilders | `torch/_utils.py:230`, `:253`, `:518`, `:354-528` |
| `TypedStorage._pickle_storage_type`, dtype/storage maps, `_LegacyStorage` | `torch/storage.py:1251`, `:548-572`, `:1541` |
| `Parameter.__new__`, `Parameter.__reduce_ex__` | `torch/nn/parameter.py:51`, `:88` |
| `Module.__init__`, `__setattr__`, `__getstate__`/`__setstate__` | `torch/nn/modules/module.py:482`, `:1976`, `:1924-1929` |
| `state_dict`, `_save_to_state_dict`, `load_state_dict`, `_load_from_state_dict` | `torch/nn/modules/module.py:2199`, `:2148`, `:2535`, `:2350` |
| Zip container reader/writer (miniz) | `caffe2/serialize/inline_container.cc` (`init` :142, `getRecord` :370, `getRecordOffset` :622, `writeRecord` :773, `writeEndOfFile` :828, `writeSerializationId` :904, padding :262-303) |
| Format version constants | `caffe2/serialize/versions.h:7-9`, `:81` |
| Python bindings `PyTorchFileWriter` / `PyTorchFileReader`, `BufferAdapter` | `torch/csrc/jit/python/init.cpp:1402`, `:1593`, `:1530` |
| `from_file` (mmap), `byteswap` | `torch/csrc/StorageMethods.cpp:410`, `:618`; `aten/src/ATen/MapAllocator.cpp:349-351` |
| Legacy raw storage I/O | `torch/csrc/serialization.cpp:235`, `:323` |
| Official notes | `docs/source/notes/serialization.md` |
