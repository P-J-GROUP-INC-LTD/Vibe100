# syntax=docker/dockerfile:1
#
# Strata: Qwen3.8-Flash-Next on NVIDIA GPUs (RTX 30/40/50, 12+ GB VRAM; two or
# three cards can share one model, 8 GB each - docs/MULTI_GPU.md).  A V100 / Titan
# V (Volta) is a build argument away - see "V100" below and docs/volta/VOLTA.md.
#
# The engine is compiled during docker build, so the first container start only
# downloads the model (~70 GB) and starts the server. docker build has no GPU,
# so the CUDA architectures are fixed here instead of read from nvidia-smi: the
# engine is a fat binary with a cubin per listed arch, and the runtime picks the
# one matching your card. Narrow CUDA_ARCHITECTURES to your card for a faster
# build; a card outside the set needs a rebuild with its own arch.
#
# Build:
#   docker build -t strata .
#   docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .        # RTX 40 only
#
# V100 / Titan V (Volta, sm_70): CUDA 13 dropped Volta, so the image needs a CUDA 12
# base - BASE_IMAGE:
#   docker build -t vibe100 \
#     --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04 .
# With no CUDA_ARCHITECTURES the list follows the toolkit: 70;75;80;86;89 on a CUDA 12
# base (a V100 and RTX 20/30/40 / A-series cards; never 120 - an RTX 50 engine built
# with CUDA 12.8 crashed on long prompts, issues #220 / #224, and one engine cannot be
# made for both), 75;80;86;89;120 on the default CUDA 13 base (CMake refuses sm_70
# there and says why).  Name the architecture to narrow it (CUDA_ARCHITECTURES=70 for a
# V100 only) or to widen it: a V100 together with an RTX 50 card in one image is not
# possible (the build stops and says so).  Run the V100 image like any other; the
# container's driver check is setup.py's (a V100 needs driver 570 or newer - CUDA 12.8 -
# and the 580 branch is the last that supports Volta, R590 does not).
#
# A card the engine has no code for (not in the list: an H100, say) makes the container
# compile one at its first start (setup.py does that, 15-20 minutes) - into the
# container's own file system, not /data: a container started with --rm, or recreated,
# compiles it again every time.  Build the image for the card's architecture instead
# (--build-arg CUDA_ARCHITECTURES=...), or keep the container (docker start) instead of
# removing it.
#
# Run (host needs an NVIDIA driver >= 580 - >= 570 for the V100 image - and
# nvidia-container-toolkit):
#   docker run --rm --gpus all \
#     -p 8080:8080 \
#     --ulimit memlock=-1 \
#     -v strata-data:/data \
#     -e MODEL=IQ2_XS \
#     strata
#
# Setup choices are env vars, read by docker-entrypoint.sh: FAMILY, MODEL, CONTEXT,
# VISION (no | yes | cpu), KV (int8 | q4_0 | k8v4), GPU (one card) or GPUS ("0,2"
# or "all", with LAYER_SPLIT), LOW_RAM (auto | on | off), HOST, PORT, API_KEY.
#
# Only the model files, the prepared pack, the MTP layer and the install config
# live in the /data volume; the engine is part of the image. Strata loads 32-62 GB
# into RAM, so a capped container needs -e LOW_RAM=on: setup.py reads the RAM from
# /proc/meminfo, which here is the host's total, not the container's limit. Add an
# API key before exposing the port to a network: -e API_KEY=<secret>. Pass
# -e REINSTALL=1 to change the model settings later.
#
# --gpus all on a host with two usable cards: setup takes both (the layer split is
# its recommended default). Pin one card with -e GPU=0, or name them with
# -e GPUS=0,2. A volume set up for one card switches to the pair on its first start
# on a two-card host unless GPU or GPUS pins it. LOW_RAM=on runs on one card.

# The CUDA toolkit the engine is compiled with.  13.0 for RTX cards (the default);
# a Volta engine needs a 12.x one (--build-arg BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04).
ARG BASE_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# STRATA_EXECV=1: setup.py replaces itself with the server, so the server is PID 1
# and docker stop's SIGTERM reaches it (see setup.start). Normal Linux starts, which
# don't set it, keep spawning the server as a child.
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 \
        python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY . .

# Empty (the default): the list follows the toolkit that is installed - 75;80;86;89;120
# with CUDA 13 (RTX 20 / A-series / RTX 30 / RTX 40 / RTX 50), 70;75;80;86;89 with CUDA 12
# (the same without the RTX 50, which needs CUDA 13, plus the V100 / Titan V, which
# CUDA 13 dropped).  CMakeLists refuses anything below 70; 70 needs a CUDA 12 BASE_IMAGE
# and cannot be in the same image as 120.  BUILD_VISION=0 skips the image encoder build.
ARG CUDA_ARCHITECTURES=""
ARG BUILD_VISION=1

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && chmod +x setup.sh docker-entrypoint.sh

# llama.cpp at the pinned commit, then the engine and the image encoder, built
# exactly the way setup.py builds them. BUILD.json is what setup.py reads to
# decide whether an engine is current: source=local with a matching src hash
# means the first start reuses it instead of recompiling.
RUN .venv/bin/python - <<'PYEOF'
import json, os, pathlib, shutil
import setup

llama = setup.get_llama_cpp()
arch = os.environ.get("CUDA_ARCHITECTURES", "").strip().strip('"').replace(",", ";")
if not arch:                             # none given: the list follows the toolkit (never 120 with CUDA 12: #220 / #224)
    arch = "70;75;80;86;89" if (setup.find_nvcc()[1] or (13, 0))[0] < 13 else "75;80;86;89;120"
archs = [int(a.split("-")[0]) for a in arch.split(";") if a.split("-")[0].isdigit()]
setup.refuse_conflict(archs)             # a Volta card and an RTX 50 cannot share one engine (CUDA 12 vs 13)
nvcc, cuda_v = setup.find_nvcc(archs)    # for Volta: only a CUDA 12.x toolkit counts
if nvcc is None:
    raise SystemExit(f"CUDA_ARCHITECTURES={arch} needs a CUDA 12.x toolkit (CUDA 13 dropped Volta, sm_70), and this "
                     "base image has none: --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04")
if setup.needs_cuda12(archs) and cuda_v < setup.CUDA12_MIN_TOOLKIT:
    raise SystemExit(f"CUDA {cuda_v[0]}.{cuda_v[1]} is too old for sm_70 on this base image (its nvcc rejects g++ 13): "
                     "--build-arg BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04")
print(f"building with CUDA {cuda_v[0]}.{cuda_v[1]} ({nvcc}) for sm_" + ", sm_".join(str(a) for a in archs))
vision = "gpu" if os.environ.get("BUILD_VISION", "1") == "1" else "none"

setup.cmake_build(setup.ROOT, setup.ROOT / "build", "strata",
    ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
     f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
     f"-DSTRATA_GGML_DIR={llama}"], None, "build-strata.bat")
if vision != "none":
    setup.cmake_build(setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision", "strata-vision",
        [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON",
         f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}"], None, "build-vision.bat")

eng = setup.ROOT / "engine"
eng.mkdir(exist_ok=True)
shutil.copy2(setup.ROOT / "build" / setup.EXE, eng / setup.EXE)
if vision != "none":
    shutil.copy2(setup.ROOT / "build-vision" / "bin" / setup.VEXE, eng / setup.VEXE)
bindir = pathlib.Path(nvcc).parent
archs = sorted(set(archs))
meta = {"source": "local", "version": setup.source_version(), "archs": archs, "vision": vision,
        "cuda": f"{cuda_v[0]}.{cuda_v[1]}",
        "cuda_dirs": [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()],
        "src": setup.source_hash(setup.ENGINE_SOURCES),
        "vision_src": setup.source_hash(setup.VISION_SOURCES) if vision != "none" else None,
        "vision_archs": archs if vision != "none" else None}
(eng / "BUILD.json").write_text(json.dumps(meta, indent=1))
PYEOF

# the cmake trees are build-time only; the engine itself is what the container needs
RUN rm -rf build build-vision

VOLUME ["/data"]
EXPOSE 8080

# /health is answered before the API key gate, so it works with or without one.
# The port only opens after the model loads (1-3 minutes, longer on a first run),
# so the start period is generous: a too short one marks a still-loading container
# unhealthy and a restart policy would kill it mid-download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
