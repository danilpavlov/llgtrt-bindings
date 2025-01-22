FROM nvcr.io/nvidia/tensorrt:24.12-py3

WORKDIR /workspaces/llgtrt-bindings

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache \
    cmake curl libjpeg-dev libpng-dev strace \
    llvm-dev libclang-dev clang ccache apache2-utils git-lfs \
    screen bsdmainutils pip python3-dev python3-venv python-is-python3 \
    pkg-config software-properties-common linux-tools-common

RUN cd /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/ && \
    ln -s libnvinfer_plugin_tensorrt_llm.so libnvinfer_plugin_tensorrt_llm.so.10

# dial down start banner a bit
RUN rm -f /opt/nvidia/entrypoint.d/40-tensorrt-samples-info.txt
# RUN rm -f /opt/nvidia/entrypoint.d/10-banner.sh
RUN rm -f /opt/nvidia/entrypoint.d/29-tensorrt-url.txt

RUN cd /usr/local/cuda/lib64 && ln -s stubs/libnvidia-ml.so libnvidia-ml.so.1

# remove stub just in case
RUN rm /usr/local/cuda/lib64/libnvidia-ml.so.1

RUN pip install pybind11
RUN export pybind11_DIR=$(python -m pybind11 --cmakedir)






