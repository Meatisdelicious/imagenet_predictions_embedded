### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Evaluation](#evaluation)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation

1. Environment requirement
    - numpy, Pillow, tqdm ...
    - vai_q_pytorch(Optional, required by quantization)
    - XIR Python frontend (Optional, required by quantization)

2. Installation with GPU Docker
   - Please refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) for how to obtain the GPU docker image.
   
   - Install all the python dependencies:
        ```shell
        pip install --user -r requirements_docker.txt
        ```

3. Installation without GPU Docker (skip this step if you are in the released Docker env)
   - Create virtual envrionment and activate it:
   ```shell
   conda create -n torch_resnet50 python=3.7
   conda activate torch_resnet50
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install --user -r requirements.txt
   ```

### Preparation

  ImageNet dataset link: [ImageNet](http://image-net.org/download-images)

  ```
  The downloaded ImageNet dataset needs to be organized as the following format:
    1) Create a folder of "data" . Put the validation data in +data/val.
    2) Data from each class for validation set needs to be put in the folder:
    +data/Imagenet/val
         +val/n01847000
         +val/n02277742
         +val/n02808304
         +...
  ```


### Evaluation
1. Evaluate float model
  ```shell
  cd code
  sh run_test_float.sh
  ```
2. Evaluate quantized(INT8) model
  ```shell
  sh run_test_quantized.sh
  ```

### Performance

We evaluate the pytorch float/quantized model accuracy:

|Model |input_size|FLOPs|Params|Float top-1/top-5 acc(%)| Quantized top-1/top-5 acc(%)|
|----|---|---|---|---|---|
|Baseline| 224x224 | 8.2G| 25.56M | 76.1/92.9| 76.1/92.9 |
|Prune0.3| 224x224 | 5.8G| 21.72M | 76.0/92.9| 75.6/92.8 |
|Prune0.4| 224x224 | 4.9G| 19.36M | 75.5/92.6| 75.1/92.5 |
|Prune0.5| 224x224 | 4.1G| 17.17M | 74.8/92.1| 74.5/92.0 |
|Prune0.6| 224x224 | 3.3G| 14.49M | 74.2/91.7| 73.9/91.5 |
|Prune0.7| 224x224 | 2.5G| 11.05M | 72.6/90.8| 72.1/90.6 |


### Model_info

1.  data preprocess
  ```
  data channel order: BGR(0~255)
  resize: short side reisze to 256 and keep the aspect ratio
  center crop: 224 * 224
  ```
