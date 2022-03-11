#!/usr/bin/env python
# coding: utf-8

#get_ipython().system('pip install --user dgl')
#get_ipython().system('pip install --user --force https://github.com/chengs/tqdm/archive/colab.zip')
#get_ipython().system('pip install --user tensorboardX')

import math
import numpy as np
import torch

from torch.utils import data
import torch.optim as optim

import math_dataset
from math_dataset import MathDatasetManager

from transformer.Models import Transformer
from math_dataset import (
    random_split_dataset,
    GraphCollate,
    MAX_ANSWER_SZ, MAX_QUESTION_SZ
)
import model_process
import utils
from tensorboard_utils import Tensorboard
from tensorboard_utils import tensorboard_event_accumulator

import checkpoints

from dgl_transformer.dgl_transformer import Transformer
from dgl_transformer.dataset.graph import GraphPool
import dgl_model_process

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
print("Torch Version", torch.__version__)



seed = 1
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


# ## Initializing Experiment
exp_name = "tmp_trial"
unique_id = "2022"


# ## Initializing Math Dataset Manager & Train/Val/Interpolate Datasets

mdsmgr = MathDatasetManager(
  "mathematics_dataset-v1.0"
)

ds = mdsmgr.build_dataset_from_module(
    'probability', 'swr_p_sequence', 'train-hard'
)
print("train-easy dataset size", len(ds))

ds_interpolate = mdsmgr.build_dataset_from_module(
    'probability', 'swr_p_sequence', 'interpolate'
)
print("interpolate dataset size", len(ds_interpolate))


# ## Build Model and Optimizer

model = utils.build_dgl_transformer()

optimizer = optim.Adam(model.parameters(), lr=6e-6, betas=(0.9, 0.995), eps=1e-9)


# ## Build Dataloaders

# Graph Collate allows to build graph batches for data_loader collation from raw math_dataset data
graph_collate = GraphCollate()


# here we split data in 80/10% for train/validation and use interpolate for test
train_ds, val_ds = math_dataset.random_split_dataset(ds, split_rate=0.8)

# all questions/answers into transformer format enhanced with char positioning
train_loader = data.DataLoader(
    train_ds, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=graph_collate(device), pin_memory=True)

val_loader = data.DataLoader(
    val_ds, batch_size=8, shuffle=False, num_workers=4,
    collate_fn=graph_collate(device))

interpolate_loader = data.DataLoader(
    ds_interpolate, batch_size=4, shuffle=False, num_workers=4,
    collate_fn=graph_collate(device))


# ## Training

tb = Tensorboard(exp_name, unique_name=unique_id)

model = model.to(device)

dgl_model_process.train(
    exp_name, unique_id,
    model, 
    train_loader, val_loader, interpolate_loader,
    optimizer, device,
    graph_pool=None,
    epochs=10, tb=tb, log_interval=100,
    start_epoch=1,
)


# ## Training remarks
# ## Training Results

# ### Extract data from Tensorboard logs

import os
path = f"./runs/{exp_name}_{unique_id}_train/"
datanames = os.listdir(path)


dgl_transfo_train_ea = tensorboard_event_accumulator(
    f"./runs/{exp_name}_{unique_id}_train/" + datanames[0]
)

path = f"./runs/{exp_name}_{unique_id}_eval/"
datanames = os.listdir(path)

dgl_transfo_valid_ea = tensorboard_event_accumulator(
    f"./runs/{exp_name}_{unique_id}_eval/" + datanames[0]
)

path = f"./runs/{exp_name}_{unique_id}_interpolate/"
datanames = os.listdir(path)

dgl_transfo_interpolate_ea = tensorboard_event_accumulator(
    f"./runs/{exp_name}_{unique_id}_interpolate/" + datanames[0]
)

dgl_transfo_train_accuracy = dgl_transfo_train_ea.Scalars("epoch/accuracy")
dgl_transfo_train_loss_per_char = dgl_transfo_train_ea.Scalars("epoch/loss_per_char")

dgl_transfo_valid_accuracy = dgl_transfo_valid_ea.Scalars("epoch/accuracy")
dgl_valid_loss_per_char = dgl_transfo_valid_ea.Scalars("epoch/loss_per_char")

dgl_transfo_interpolate_accuracy = dgl_transfo_interpolate_ea.Scalars("epoch/accuracy")
dgl_transfo_interpolate_loss_per_char = dgl_transfo_interpolate_ea.Scalars("epoch/loss_per_char")


# ### Training Logs

plt.rcParams['figure.figsize'] = [10, 6]

fig, ax = plt.subplots()


ax.plot(
    list(map(lambda l: l.step, dgl_transfo_train_accuracy)),
    list(map(lambda l: l.value, dgl_transfo_train_accuracy)),
    marker='o', label='DGL Transfo Train Accuracy'
)
plt.title(exp_name + ' Train Accuracy')
ax.legend(loc='upper left', frameon=False)
plt.xticks(np.arange(0, 15, step=1.0))
plt.yticks(np.arange(0.3, 1.0, step=0.1))
plt.savefig(exp_name + '_dgl_train_Accuracy.jpg')
plt.show()


plt.rcParams['figure.figsize'] = [10, 6]

fig, ax = plt.subplots()


ax.plot(
    list(map(lambda l: l.step, dgl_transfo_valid_accuracy)),
    list(map(lambda l: l.value, dgl_transfo_valid_accuracy)),
    marker='o', label='DGL Transfo Validation Accuracy'
)
ax.plot(
    list(map(lambda l: l.step, dgl_transfo_interpolate_accuracy)),
    list(map(lambda l: l.value, dgl_transfo_interpolate_accuracy)),
    marker='o', label='DGL Transfo Interpolate Accuracy', color="orange"
)
plt.title(exp_name + ' Accuracy')
ax.legend(loc='upper left', frameon=False)
plt.xticks(np.arange(0, 11, step=1.0))
plt.yticks(np.arange(0.3, 1.0, step=0.05))
plt.savefig(exp_name + '_dgl_Accuracy.jpg')
plt.show()

print(max(dgl_transfo_valid_accuracy))
print(max(dgl_transfo_interpolate_accuracy))


# ## Check Inference

device = torch.device("cpu")
# build default transformer model
model = utils.build_dgl_transformer()
# restore model from checkpoint
_ = checkpoints.restore_best_checkpoint(exp_name, unique_id, "validation", model)
model = model.to(device)


print("What is 1 + 1", dgl_model_process.predict_single("What is 1 + 1", model, device, graph_collate))

print("What is 10 + 10", dgl_model_process.predict_single("What is 10 + 10", model, device, graph_collate))








