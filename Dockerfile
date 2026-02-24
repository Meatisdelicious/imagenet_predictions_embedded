FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl bzip2 ca-certificates git rsync openssh-client \
 && rm -rf /var/lib/apt/lists/*

ENV HOME=/workspace
WORKDIR /workspace
RUN groupadd -g 42420 ovh || true \
 && useradd -u 42420 -g 42420 -d /workspace -s /bin/bash -M ovh || true \
 && chown -R 42420:42420 /workspace

USER ovh
SHELL ["/bin/bash", "--login", "-c"]
