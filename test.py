# Copyright 2024 The Flax Authors.
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

"""MNIST example.

Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""
from main import get_train_dataloader
import jax


jax.distributed.initialize()

EPOCHS = 1000  # @param{type:"integer"}
# @markdown Number of samples for each batch in the training set:
TRAIN_BATCH_SIZE = 64  # @param{type:"integer"}
# @markdown Number of samples for each batch in the test set:
TEST_BATCH_SIZE = 64  # @param{type:"integer"}
# @markdown Learning rate for the optimizer:
LEARNING_RATE = 0.001  # @param{type:"number"}
# @markdown The dataset to use.
DATASET = "cifar10"  # @param{type:"string"}
# @markdown The amount of L2 regularization to use:
L2_REG = 0.0001  # @param{type:"number"}
# @markdown Adversarial perturbations lie within the infinity-ball of radius epsilon.
EPSILON = 8 / 255  # @param{type:"number"}

os.environ['WANDB_API_KEY'] = 'ec6aa52f09f51468ca407c0c00e136aaaa18a445'

if __name__ == "__main__":
    _, train_dataloader, test_dataloader = get_train_dataloader(TRAIN_BATCH_SIZE)
    train_dataloader_iter = iter(train_dataloader)

    for _ in range(100):
        data = next(train_dataloader_iter)

    # train_and_evaluate()
