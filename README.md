# ML model loading deep dives

How do ML frameworks structure their model files, and what exactly happens, in the source code, when a model is loaded? One directory per framework, each with a written deep dive, a text flow chart of the load path, a structure figure, and a small script that reproduces every claim.

| Framework / format | Deep dive | Status |
|---|---|---|
| PyTorch (`torch.save` / `torch.load`, `.pt` / `.pth`) | [pytorch/README.md](pytorch/README.md) | done (torch 2.14.0) |
| Keras v3 (`.keras`) | | planned |
| safetensors | | planned |

Conventions, method and the framework backlog live in [CLAUDE.md](CLAUDE.md). Shared figure tooling is in [`tools/`](tools/).
