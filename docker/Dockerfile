ARG BUILD_TARGET=embodied
ARG TARGETARCH
# Hardware platform for the install: nvidia (default), amd (ROCm), ascend (CANN),
# or musa (Moore Threads).
ARG PLATFORM=nvidia
# Per-platform runtime versions. Override these for common CUDA/ROCm/CANN bumps, or
# override the full base-image args below for a different image family/Ubuntu.
ARG CUDA_VER=12.8.1
ARG ROCM_VER=6.4
ARG ROCM_ARCHS=gfx90a;gfx942
ARG CANN_VER=9.0.0-910b
ARG MUSA_VER=v2.1.5-musa4.3.7
ARG UBUNTU_VER=22.04
# Registry prefix for the Docker Hub base images, with a trailing slash.
# Empty pulls from Docker Hub; set e.g. REGISTRY_MIRROR=m.daocloud.io/docker.io/ when it is unreachable.
ARG REGISTRY_MIRROR=
ARG NVIDIA_BASE_IMAGE=${REGISTRY_MIRROR}nvidia/cuda:${CUDA_VER}-cudnn-devel-ubuntu${UBUNTU_VER}
ARG AMD_BASE_IMAGE=${REGISTRY_MIRROR}rocm/dev-ubuntu-${UBUNTU_VER}:${ROCM_VER}-complete
# Ascend lives on Huawei SWR, not Docker Hub.
ARG ASCEND_BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${CANN_VER}-ubuntu${UBUNTU_VER}-py3.11
# MUSA lives on the Moore Threads registry, not Docker Hub.
ARG MUSA_BASE_IMAGE=registry.mthreads.com/mcctest/ai/training-suite:${MUSA_VER}
ARG FRANKA_BASE_IMAGE=${REGISTRY_MIRROR}ubuntu:20.04
# Existing RLinf ManiSkill/LIBERO image used only by the PhyAI integration target.
ARG RLINF_PHYAI_BASE_IMAGE=rlinf/rlinf:agentic-rlinf0.3-maniskill_libero


##################################################################################################
# Platform base resolution (NVIDIA, AMD/ROCm, Ascend/CANN, Moore Threads/MUSA)
##################################################################################################
FROM ${NVIDIA_BASE_IMAGE} AS base-image-platform-nvidia
FROM ${AMD_BASE_IMAGE} AS base-image-platform-amd
ARG ROCM_ARCHS
ENV AMDGPU_TARGETS=${ROCM_ARCHS}
ENV GPU_ARCHS=${ROCM_ARCHS}
ENV HCC_AMDGPU_TARGET=${ROCM_ARCHS}
ENV PYTORCH_ROCM_ARCH=${ROCM_ARCHS}
ENV AMD_VISIBLE_DEVICES=all
FROM ${ASCEND_BASE_IMAGE} AS base-image-platform-ascend
ENV ASCEND_VISIBLE_DEVICES=all
FROM ${MUSA_BASE_IMAGE} AS base-image-platform-musa
ENV MTHREADS_VISIBLE_DEVICES=all
ENV MTHREADS_DRIVER_CAPABILITIES=all

##################################################################################################
# Base images for different build targets — all derive from the platform base
##################################################################################################
FROM base-image-platform-${PLATFORM} AS base-image-reason
FROM base-image-platform-${PLATFORM} AS base-image-embodied-maniskill_libero
FROM base-image-platform-${PLATFORM} AS base-image-embodied-behavior-openvlaoft
FROM base-image-platform-${PLATFORM} AS base-image-embodied-behavior-openpi
FROM base-image-platform-${PLATFORM} AS base-image-embodied-metaworld
FROM base-image-platform-${PLATFORM} AS base-image-embodied-calvin
FROM base-image-platform-${PLATFORM} AS base-image-embodied-robocasa
FROM base-image-platform-${PLATFORM} AS base-image-embodied-robocasa365
FROM base-image-platform-${PLATFORM} AS base-image-embodied-isaaclab
FROM base-image-platform-${PLATFORM} AS base-image-embodied-robotwin
FROM base-image-platform-${PLATFORM} AS base-image-embodied-opensora
FROM base-image-platform-${PLATFORM} AS base-image-embodied-frankasim
FROM base-image-platform-${PLATFORM} AS base-image-embodied-wan
FROM base-image-platform-${PLATFORM} AS base-image-embodied-libero
FROM base-image-platform-${PLATFORM} AS base-image-embodied-liberopro
FROM base-image-platform-${PLATFORM} AS base-image-embodied-liberoplus
FROM base-image-platform-${PLATFORM} AS base-image-embodied-embodichain
FROM base-image-platform-${PLATFORM} AS base-image-embodied-roboverse
FROM base-image-platform-${PLATFORM} AS base-image-embodied-polaris
FROM base-image-platform-${PLATFORM} AS base-image-embodied-genesis


# Franka depends on ros-noetic (Ubuntu 20.04) and is GPU-agnostic, so it
# bypasses the platform switch and always uses a plain Ubuntu base.
FROM ${FRANKA_BASE_IMAGE} AS base-image-embodied-franka


##################################################################################################
# Select base image according to BUILD_TARGET
##################################################################################################
FROM base-image-$BUILD_TARGET AS base-image
ARG PLATFORM
ENV RLINF_PLATFORM=${PLATFORM}


##################################################################################################
# System-level setup (apt, mirrors, basic tools)
##################################################################################################
ARG TARGETARCH
ARG NO_MIRROR
SHELL ["/bin/bash", "-c"]
ENV NO_MIRROR=${NO_MIRROR}
ENV DEBIAN_FRONTEND=noninteractive
RUN if [ "$NO_MIRROR" != "1" ]; then \
        if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
            if [ "$TARGETARCH" = "arm64" ]; then \
                sed -i 's@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources; \
            else \
                sed -i \
                    -e 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' \
                    -e 's@//security.ubuntu.com@//mirrors.ustc.edu.cn@g' \
                    /etc/apt/sources.list.d/ubuntu.sources; \
            fi; \
        elif [ "$TARGETARCH" = "arm64" ]; then \
            sed -i \
                -e 's@//ports.ubuntu.com/\? @//ports.ubuntu.com/ubuntu-ports @g' \
                -e 's@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g' \
                /etc/apt/sources.list; \
        else \
            sed -i \
                -e 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' \
                -e 's@//security.ubuntu.com@//mirrors.ustc.edu.cn@g' \
                /etc/apt/sources.list; \
        fi; \
    fi
RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim libibverbs-dev openssh-server sudo runit runit-systemd tmux \
    build-essential python3-dev cmake pkg-config iproute2 pciutils python3 python3-pip \
    wget unzip curl


##################################################################################################
# Python / pip / uv setup
##################################################################################################

ENV PIP_INDEX=${NO_MIRROR:+"https://pypi.org/simple/"}
ENV PIP_INDEX=${PIP_INDEX:-"https://mirrors.ustc.edu.cn/pypi/simple"}
ENV UV_DEFAULT_INDEX=${NO_MIRROR:+"https://pypi.org/simple/"}
ENV UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-"https://mirrors.ustc.edu.cn/pypi/simple"}
ENV INSTALL_MIRROR_OPTION=${NO_MIRROR:+" "}
ENV INSTALL_MIRROR_OPTION=${INSTALL_MIRROR_OPTION:-"--use-mirror"}
ENV HF_ENDPOINT=${NO_MIRROR:+"https://huggingface.co"}
ENV HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

RUN pip config set global.index-url ${PIP_INDEX}
RUN python3 -m pip install -i ${PIP_INDEX} --upgrade pip setuptools wheel uv

ENV HF_HOME=/opt/.cache/huggingface
RUN mkdir -p $HF_HOME

ENV UV_PATH=/opt/venv
RUN mkdir $UV_PATH
WORKDIR $UV_PATH

ENV UV_LINK_MODE=hardlink
ENV UV_CACHE_DIR=$UV_PATH/.cache
ENV UV_PYTHON_INSTALL_DIR=${UV_PATH}/.python


##################################################################################################
# Project requirements
##################################################################################################
COPY ./pyproject.toml $UV_PATH/pyproject.toml
COPY ./requirements $UV_PATH/requirements


##################################################################################################
# Utilities: switch_env
##################################################################################################
RUN cat <<EOF > /usr/local/bin/switch_env
#!/bin/bash
if [ -z "\$1" ]; then
    echo "Usage: switch_env <env_name>"
    exit 1
fi
if [ ! -d "${UV_PATH}/\$1" ]; then
    echo "Environment \$1 does not exist in ${UV_PATH}."
    exit 1
fi
source ${UV_PATH}/\$1/bin/activate
EOF
RUN chmod +x /usr/local/bin/switch_env


##################################################################################################
# Reason image
##################################################################################################
FROM base-image AS reason-image

# One venv per rollout engine: sglang and vllm pin the same kernel libraries
# (nvidia-cutlass-dsl, flashinfer, tilelang, tokenspeed-mla) to different
# versions, so sharing a venv means whichever is installed second downgrades the
# other's kernels. Activate `reason-vllm` for vLLM rollout.
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} agentic --venv reason --engine sglang && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} agentic --venv reason-vllm --engine vllm

# Set default env
RUN echo "source ${UV_PATH}/reason/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied common base (shared by all embodied environments)
##################################################################################################
FROM base-image AS embodied-common-image

# Embodied NVIDIA_DRIVER_CAPABILITIES
ENV NVIDIA_DRIVER_CAPABILITIES="all"


##################################################################################################
# Utilities: asset linking & downloading
##################################################################################################
RUN cat <<EOF > /usr/local/bin/link_assets
#!/bin/bash
if [ -d /opt/assets/.maniskill ]; then
    if [ -d ~/.maniskill ]; then
        rm -rf ~/.maniskill
    fi
    ln -s /opt/assets/.maniskill ~/.maniskill
fi
if [ -d /opt/assets/.sapien ]; then
    if [ -d ~/.sapien ]; then
        rm -rf ~/.sapien
    fi
    ln -s /opt/assets/.sapien ~/.sapien
fi
mkdir -p ~/.cache
if [ -d /opt/assets/.cache/openpi ]; then
    if [ -d ~/.cache/openpi ]; then
        rm -rf ~/.cache/openpi
    fi
    ln -s /opt/assets/.cache/openpi ~/.cache/openpi
fi
EOF
RUN chmod +x /usr/local/bin/link_assets

RUN cp $UV_PATH/requirements/embodied/download_assets.sh /usr/local/bin/download_assets
RUN chmod +x /usr/local/bin/download_assets


##################################################################################################
# Embodied: ManiSkill / Libero
##################################################################################################
FROM embodied-common-image AS embodied-maniskill_libero-image

# Install openvla env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla --model openvla --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv gr00t --model gr00t --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv gr00t_n1d6 --model gr00t_n1d6 --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv gr00t_n1d7 --model gr00t_n1d7 --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv dexbotic --model dexbotic --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv starvla --model starvla --env maniskill_libero && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv abot_m0 --model abot_m0 --env maniskill_libero

RUN source switch_env openvla && download_assets ${INSTALL_MIRROR_OPTION} --dir /opt/assets --assets maniskill,openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openvla/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: ManiSkill / Libero + PhyAI (isolated Python 3.12 runtimes)
##################################################################################################
FROM ${RLINF_PHYAI_BASE_IMAGE} AS embodied-maniskill_libero-phyai-image

ARG RLINF_PHYAI_PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple
ENV UV_DEFAULT_INDEX=${RLINF_PHYAI_PYPI_INDEX}
ENV PIP_INDEX_URL=${RLINF_PHYAI_PYPI_INDEX}
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_XET=1
ENV HF_HOME=/opt/venv/.cache/huggingface
ENV UV_LINK_MODE=copy

COPY . /opt/src/RLinf
COPY --from=phyai . /opt/src/phyai

# Git checkouts on the build host may use CRLF; container shell scripts need LF.
RUN find /opt/src/RLinf /opt/src/phyai -type f -name '*.sh' \
        -exec sed -i 's/\r$//' {} +

# These source patches belong only to the integration stage. Keep existing
# RLinf stages on transformers 4.x and make LIBERO assets real venv files
# instead of symlinks into the temporary Hugging Face cache. Reuse the exact
# ManiSkill version already present in the immutable parent image.
RUN sed -i 's/transformers<=4\.57\.6/transformers==5.8.1/' /opt/src/RLinf/pyproject.toml && \
    sed -i 's/retry_cmd libero-download-assets --skip-existing/true # deferred to a cacheable integration layer/' \
        /opt/src/RLinf/requirements/install.sh && \
    sed -i '/ManiSkill\.git@v3\.0\.0b22/c\    true # copied from the immutable base image' \
        /opt/src/RLinf/requirements/install.sh && \
    sed -i '/download_assets\.sh --assets maniskill/c\    true # included in the base image package' \
        /opt/src/RLinf/requirements/install.sh && \
    grep -q 'copied from the immutable base image' /opt/src/RLinf/requirements/install.sh && \
    grep -q 'transformers==5.8.1' /opt/src/RLinf/pyproject.toml

# Install RLinf core and the ManiSkill/LIBERO runtime without the RLinf OpenPI
# package, whose patched transformers 4.x files cannot share this environment.
RUN --mount=type=cache,target=/opt/venv/.cache \
    cd /opt/src/RLinf && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} \
        --platform ${RLINF_PLATFORM} \
        embodied \
        --venv /opt/venv/rlinf-phyai \
        --python 3.12 \
        --torch 2.11.0 \
        --env maniskill_libero \
        --no-root \
        --install-rlinf

# ManiSkill 3.0.0b22 is pure Python; copy the exact distribution already
# installed in the pinned parent image, leaving that source venv untouched.
RUN cp -a /opt/venv/abot_m0/lib/python3.11/site-packages/mani_skill \
        /opt/venv/rlinf-phyai/lib/python3.12/site-packages/ && \
    cp -a /opt/venv/abot_m0/lib/python3.11/site-packages/mani_skill-3.0.0b22.dist-info \
        /opt/venv/rlinf-phyai/lib/python3.12/site-packages/ && \
    /opt/venv/rlinf-phyai/bin/python -c "import importlib.metadata as m; assert m.version('mani_skill') == '3.0.0b22'"

# Keep partial Hugging Face downloads across retries, then copy the completed
# assets into the venv so the final image contains no cache symlink.
RUN --mount=type=cache,target=/var/cache/rlinf-libero-assets \
    source /opt/venv/rlinf-phyai/bin/activate && \
    libero-download-assets \
        --assets-dir /var/cache/rlinf-libero-assets \
        --force \
        --no-cache && \
    rm -rf /opt/venv/rlinf-phyai/lib/python3.12/site-packages/libero/libero/assets && \
    mkdir -p /opt/venv/rlinf-phyai/lib/python3.12/site-packages/libero/libero/assets && \
    cp -a /var/cache/rlinf-libero-assets/. /opt/venv/rlinf-phyai/lib/python3.12/site-packages/libero/libero/assets/ && \
    rm -rf /opt/venv/rlinf-phyai/lib/python3.12/site-packages/libero/libero/assets/.cache

# Preserve the RLinf runtime while resolving and installing the complete PhyAI
# workspace into the same active environment. Reconcile shared transitive
# dependencies after uv sync, including requirements declared by the copied
# ManiSkill distribution.
RUN --mount=type=cache,target=/opt/venv/.cache \
    source /opt/venv/rlinf-phyai/bin/activate && \
    cd /opt/src/phyai && \
    uv sync --active --inexact --no-dev && \
    python -c 'import importlib.metadata as m; print("\n".join(m.requires("mani_skill") or []))' \
        > /tmp/mani-skill-requirements.txt && \
    uv pip install \
        -r /tmp/mani-skill-requirements.txt \
        'numpy<2' \
        'nvidia-ml-py<13.596.0a0' \
        'fsspec[http]<=2025.3.0' \
        'ml-dtypes<0.6' \
        'opencv-python-headless<5' \
        pytest-asyncio && \
    uv pip check

# rlinf-openpi ships a complete Transformers 4.53.2 fork and overwrites the
# same package files used by stock Transformers 5.8.1. Keep it in a separate
# venv, while reusing the large, immutable Torch 2.11 and LIBERO installation
# from the PhyAI environment through a lower-priority .pth entry.
#
# rlinf-openpi 0.1.1 currently declares torch==2.7.1 and an unconstrained
# lerobot dependency. Installing it with dependencies would downgrade Torch
# and makes resolution depend on whichever lerobot release is newest. Install
# the exact RLinf distributions without dependencies, then install the
# repository's OpenPI runtime pins explicitly. Native pi0.5 inference does not
# import lerobot.
RUN --mount=type=cache,target=/opt/venv/.cache \
    uv venv \
        --python /opt/venv/rlinf-phyai/bin/python \
        /opt/venv/rlinf-openpi && \
    echo /opt/venv/rlinf-phyai/lib/python3.12/site-packages \
        > /opt/venv/rlinf-openpi/lib/python3.12/site-packages/rlinf-base.pth && \
    uv pip install \
        --python /opt/venv/rlinf-openpi/bin/python \
        'transformers==4.53.2' \
        'tokenizers>=0.21,<0.22' \
        'huggingface-hub>=0.30,<1.0' \
        'numpy<2' \
        'fsspec[http]<=2025.3.0' && \
    uv pip install \
        --python /opt/venv/rlinf-openpi/bin/python \
        --no-deps \
        'rlinf-openpi==0.1.1' \
        'rlinf-transformer-openpi==4.53.2' && \
    uv pip install \
        --python /opt/venv/rlinf-openpi/bin/python \
        -r /opt/src/RLinf/requirements/embodied/models/openpi.txt \
        'numpy<2' \
        'fsspec[http]<=2025.3.0' \
        'augmax>=0.3.4' \
        'beartype==0.19.0' \
        'chex==0.1.90' \
        'equinox>=0.11.8' \
        'flax==0.10.2' \
        'flatbuffers>=24.3.25' \
        'gym-aloha>=0.1.1' \
        'jaxtyping==0.2.36' \
        'ml-collections==1.0.0' \
        'numpydantic>=1.6.6' \
        'opencv-python==4.11.0.86' \
        'polars>=1.30.0' \
        'sentencepiece>=0.2.0' \
        'tqdm-loggable>=0.2' \
        'treescope>=0.1.7' \
        'wandb>=0.19.1' \
        openpi-client && \
    cp -r \
        /opt/venv/rlinf-openpi/lib/python3.12/site-packages/openpi/models_pytorch/transformers_replace/. \
        /opt/venv/rlinf-openpi/lib/python3.12/site-packages/transformers/ && \
    PYTHONPATH=/opt/src/RLinf /opt/venv/rlinf-openpi/bin/python -c \
        "import openpi, rlinf, torch, transformers; assert torch.__version__.startswith('2.11.0'); assert transformers.__version__ == '4.53.2'" && \
    /opt/venv/rlinf-phyai/bin/python -c \
        "import phyai, torch, transformers; assert torch.__version__.startswith('2.11.0'); assert transformers.__version__ == '5.8.1'" && \
    ln -s /opt/venv/rlinf-phyai /opt/venv/phyai

ENV PHYAI_VIRTUAL_ENV=/opt/venv/phyai
ENV RLINF_OPENPI_VIRTUAL_ENV=/opt/venv/rlinf-openpi

ENV VIRTUAL_ENV=/opt/venv/rlinf-phyai
ENV PATH="/opt/venv/rlinf-phyai/bin:${PATH}"
ENV PYTHONPATH=/opt/src/RLinf
ENV REPO_PATH=/opt/src/RLinf
ENV EMBODIED_PATH=/opt/src/RLinf/examples/embodiment

# The parent stage activates openvla for interactive shells; make this stage's
# integration environment the final activation instead.
RUN echo "source /opt/venv/rlinf-phyai/bin/activate" >> ~/.bashrc

WORKDIR /opt/src/RLinf


FROM embodied-common-image AS embodied-behavior-openvlaoft-image

# Install openvla-oft env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env behavior

# omnigibson downloads assets to /tmp during initialization
RUN source switch_env openvla-oft && python -c "import omnigibson as og; cfg={}; og.Environment(cfg); og.shutdown()" || exit 0

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

FROM embodied-common-image AS embodied-behavior-openpi-image

# Install openpi env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env behavior

# omnigibson downloads assets to /tmp during initialization
RUN source switch_env openpi && python -c "import omnigibson as og; cfg={}; og.Environment(cfg); og.shutdown()" || exit 0

RUN source switch_env openpi && download_assets ${INSTALL_MIRROR_OPTION} --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: MetaWorld
##################################################################################################
FROM embodied-common-image AS embodied-metaworld-image

# Install openvla-oft and openpi envs
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env metaworld && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env metaworld

RUN source switch_env openpi && download_assets ${INSTALL_MIRROR_OPTION} --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: CALVIN
##################################################################################################
FROM embodied-common-image AS embodied-calvin-image

# Clone the CALVIN env repo once into a shared path; both models below reuse it.
ENV CALVIN_PATH=/opt/envs/calvin

# Install openvla-oft and openpi envs
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env calvin && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env calvin

RUN source switch_env openpi && download_assets ${INSTALL_MIRROR_OPTION} --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: RoboCasa
##################################################################################################
FROM embodied-common-image AS embodied-robocasa-image

# Install openpi env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env robocasa

RUN source switch_env openpi && download_assets ${INSTALL_MIRROR_OPTION} --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: RoboCasa365
##################################################################################################
FROM embodied-common-image AS embodied-robocasa365-image

RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env robocasa365

RUN source switch_env openpi && download_assets ${INSTALL_MIRROR_OPTION} --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: IsaacLab
##################################################################################################
FROM embodied-common-image AS embodied-isaaclab-image

# Clone the IsaacLab env repo once into a shared path; both models below reuse it.
ENV ISAAC_LAB_PATH=/opt/envs/isaaclab

# Install gr00t env, and openpi env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv gr00t --model gr00t --env isaaclab && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env isaaclab

# Set default env
RUN echo "source ${UV_PATH}/gr00t/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: Franka (Ubuntu 20.04, ROS Noetic)
##################################################################################################
FROM embodied-common-image AS embodied-franka-image

# Install multiple Franka environments with different libfranka versions
# franka uses ubuntu:20.04 with no GPU stack regardless of build-arg platform,
# so always invoke install.sh with --platform nvidia to skip ROCm detection.
RUN for v in 0.10.0 0.13.3 0.14.1 0.15.0 0.18.0 0.19.0; do \
            LIBFRANKA_VERSION="$v" FRANKA_ROS_VERSION=0.10.0 bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform nvidia embodied --venv "franka-$v" --env franka; \
        done

# Franky environments. franky-control ships libfranka inside a prebuilt wheel,
# so these need no ROS/catkin build and are kept in separate venvs from the
# franka-$v ones above. Only the libfranka versions with published wheels are
# built: https://github.com/Brunch-Life/franky/releases
RUN for v in 0.15.0 0.19.0; do \
            LIBFRANKA_VERSION="$v" bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform nvidia embodied --venv "franky-$v" --env franka-franky; \
        done

RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform nvidia embodied --venv franka-dexhand --env franka-dexhand

# Set default env (recommended libfranka 0.15.0)
RUN echo "source ${UV_PATH}/franka-0.15.0/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: RoboTwin
##################################################################################################
FROM embodied-common-image AS embodied-robotwin-image

# Install openvla-oft, openpi, and lingbotvla envs
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env robotwin && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env robotwin && \
    bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv lingbotvla --model lingbotvla --env robotwin

RUN source switch_env openpi && download_assets ${INSTALL_MIRROR_OPTION} --dir /opt/assets --assets openpi
RUN link_assets

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc


##################################################################################################
# Embodied: OpenSora
##################################################################################################
FROM embodied-common-image AS embodied-opensora-image

# Install openvla-oft env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env opensora

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: WAN
##################################################################################################
FROM embodied-common-image AS embodied-wan-image

# Install openvla-oft env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env wan

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: FrankaSim
##################################################################################################
FROM embodied-common-image AS embodied-frankasim-image

# Install openvla env
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla --model openvla --env frankasim

# Set default env
RUN echo "source ${UV_PATH}/openvla/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: EmbodiChain
##################################################################################################
FROM embodied-common-image AS embodied-embodichain-image

RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv embodichain --env embodichain

RUN echo "source ${UV_PATH}/embodichain/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: LIBERO
##################################################################################################
FROM embodied-common-image AS embodied-libero-image

# Ascend supports the openvla, openvla-oft and gr00t models only.
RUN if [ "${RLINF_PLATFORM}" = "ascend" ]; then \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla --model openvla --env libero && \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env libero && \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv gr00t --model gr00t --env libero; \
    else \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla --model openvla --env libero && \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env libero && \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv dreamzero --model dreamzero --env libero && \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv gr00t --model gr00t --env libero && \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv molmoact2 --model molmoact2 --env libero && \
        bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv evo1 --model evo1 --env libero; \
    fi

# Set default env
RUN echo "source ${UV_PATH}/openvla/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: LIBERO-Pro
##################################################################################################
FROM embodied-common-image AS embodied-liberopro-image

# Install openvla-oft env for LIBERO-Pro
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env liberopro

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: LIBERO-Plus
##################################################################################################
FROM embodied-common-image AS embodied-liberoplus-image

# Install openvla-oft env for LIBERO-Plus
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openvla-oft --model openvla-oft --env liberoplus

# Set default env
RUN echo "source ${UV_PATH}/openvla-oft/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: RoboVerse
##################################################################################################
FROM embodied-common-image AS embodied-roboverse-image

# Install openpi env for RoboVerse
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env roboverse

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: Polaris
##################################################################################################
FROM embodied-common-image AS embodied-polaris-image

# Install openpi env for Polaris
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv openpi --model openpi --env polaris

# Set default env
RUN echo "source ${UV_PATH}/openpi/bin/activate" >> ~/.bashrc

##################################################################################################
# Embodied: Genesis
##################################################################################################
FROM embodied-common-image AS embodied-genesis-image

# Install genesis env (MLP/CNN policy, no VLA model)
RUN bash requirements/install.sh ${INSTALL_MIRROR_OPTION} --platform ${RLINF_PLATFORM} embodied --venv genesis --env genesis

# Set default env
RUN echo "source ${UV_PATH}/genesis/bin/activate" >> ~/.bashrc

##################################################################################################
# Final image selection
##################################################################################################
FROM ${BUILD_TARGET}-image AS final-image


##################################################################################################
# Cleanup & runtime defaults
##################################################################################################
RUN uv cache prune
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
