# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo " Test float weight "
export W_QUANT=0
cd ../

python code/test/eval_imagenet.py \
    --model_name resnet50 \
    --data_dir data/Imagenet/ \
    --model_path float/resnet50_pretrained.pth \
    --batch_size 128 \
