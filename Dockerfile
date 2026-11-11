# syntax=docker/dockerfile:1
# https://github.com/Kaggle/docker-python/releases
FROM gcr.io/kaggle-gpu-images/python:latest

# 依存変更時もダウンロード済みパッケージを再利用する。
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --cache-dir=/root/.cache/pip \
    hydra-core onnxruntime-gpu openvino onnxscript schedulefree
