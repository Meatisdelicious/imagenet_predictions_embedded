.PHONY: install-dependencies docker-pull docker-run 

####################### to install the uv virtual env and it's dependencies :
VENV_DIR := .venv

install-dependencies: # creates uv venv and installs the dependencies in it
	@echo "1_Creating uv venv and installing deps..."
	uv venv $(VENV_DIR) && uv sync

######################## To instal and run the docker container : 

# ---- Config ----
IMAGE := xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
CONTAINER := vitis_ai_3_0_0_106
WORKDIR := /workspace
# Use host project directory
PWD_HOST := $(shell pwd)
# If you're on macOS with XQuartz, DISPLAY is usually host.docker.internal:0
# If on Linux, DISPLAY is often :0 and you mount /tmp/.X11-unix
DISPLAY ?= host.docker.internal:0
# Platform override (needed on Apple Silicon; harmless elsewhere if you remove it)
PLATFORM := --platform linux/amd64

# ---- Targets ----
docker-pull:
	@echo "📥 Pulling Docker image: $(IMAGE)"
	docker pull $(IMAGE)

docker-run: docker-pull
	@echo "🐳 Starting interactive container shell (old workflow)..."
	@echo "Inside the container, run: make docker-setup (or copy/paste the commands it prints)."
	docker run --rm -it \
	  --name $(CONTAINER) \
	  $(PLATFORM) \
	  -e DISPLAY=$(DISPLAY) \
	  -e UID=$$(id -u) -e GID=$$(id -g) \
	  -v /tmp/.X11-unix:/tmp/.X11-unix \
	  -v "$(PWD_HOST)":$(WORKDIR) \
	  -w $(WORKDIR) \
	  $(IMAGE) /bin/bash

# After the docker has started, don't forget to run those commands, inside the docker. Otherwise it won't run properly on mac
# export NNDCT_DEVICE=cpu
# Install dependencies that are not on docker image : 
# pip install torch==1.12.1 torchvision==0.13.0a0 torchsummary==1.5.1 tabulate==0.9.0 graphviz==0.20.1