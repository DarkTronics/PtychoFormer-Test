import os
import wandb
import pprint
import torch
import numpy as np
import torch.nn as nn
from Models.PtychoFormer import PtychoFormer
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_optimizer import Prodigy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

os.environ['WANDB_NOTEBOOK_NAME'] = 'train.py'
wandb.login()

num_trails = 1
sweep_config = {'method': 'random'}
metric = {'name': 'trn loss','goal': 'minimize'}
sweep_config['metric'] = metric
parameters_dict = {
    'epochs': {'value': 100},
    'class_batch': {'value': 10}
  }
sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

batch = 30
checkpoint = True
checkpoint_rate = 1

class Data(Dataset):
  def __init__(self, dir, num_data):
    self.dir = dir
    self.num_data = num_data
   
  def __len__(self):
    return self.num_data
  
  def __getitem__(self, i):
    data = torch.load(os.path.join(self.dir, f'/data_{i}'))
    diff = data['input']
    diff=diff[torch.randperm(9)]
    phase_amp = data['label']
    return (diff, phase_amp)

class GroupedSampler:
    """Sampler that groups images of the same size and samples within these groups for training."""
    def __init__(self, batch_size, train_indices, class_num):
        self.batch_size = batch_size
        self.train_indices = np.array(train_indices)
        self.num_groups = class_num
        self.cls_num_data = len(self.train_indices) // class_num
        self.num_batch = len(self.train_indices) // batch_size

    def __iter__(self):
        shuffled_indices = np.array(self.train_indices)
        # Shuffle the order of groups
        for group in range(self.num_groups):
            start_idx = group * self.cls_num_data
            end_idx = start_idx + self.cls_num_data
            np.random.shuffle(shuffled_indices[start_idx:end_idx])

        batch_order = np.random.permutation(self.num_batch)
        for batch in batch_order:
            for i in range(self.batch_size):
              yield shuffled_indices[batch*self.batch_size + i]

    def __len__(self):
        return len(self.train_indices)


def build_dataset(batch_size):
    data_file_path = os.path.join(os.getcwd(), 'dataset', 'pretrain', 'data')
    dir = os.path.normpath(data_file_path)
    data_per_class = 6800
    test_data_per_class = 680
    num_class = 36
    num_data = data_per_class * num_class
    train_data = Data(dir, num_data)
    val_data = Data(dir, num_data)
    
    val_batch_size = 10
    train_indices = []
    val_indices = []
    for i in range(0, num_data, data_per_class):  # Iterate over each class
       group_indices = range(i, i+data_per_class)
       train_indices.extend(group_indices[:-test_data_per_class])
       val_indices.extend(group_indices[-test_data_per_class:])

    print("Training Set:", len(train_indices))
    train_dataset = Subset(train_data, train_indices)
    val_dataset = Subset(val_data, val_indices)

    sampler_indices = range(0, len(train_dataset))
    train_sampler = GroupedSampler(batch_size, sampler_indices, class_num=num_class) # 36

    assert(batch_size * (len(train_dataset) // batch_size)  == len(train_dataset))
    assert(val_batch_size * (len(val_dataset) // val_batch_size) == len(val_dataset))

    trn_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    return trn_dl, val_dl

def build_network():
  model = PtychoFormer()
  model = model.to(device)
  return model

def build_optimizer(network):
  optimizer = Prodigy(network.parameters())
  return optimizer

def network_loss(pred, target, norm = True):
    l1_fn = nn.L1Loss()
    if norm:
      pred[:, 0:1, :, :] = (pred[:, 0:1, :, :] + np.pi) / (2 * np.pi)
      target[:, 0:1, :, :]  = (target[:, 0:1, :, :] + np.pi) / (2 * np.pi)
    return l1_fn(pred, target)

def train_epoch(network, loader, optimizer, data_size, epoch, accumulation_steps):
    epoch_loss = 0
    cumu_loss = 0
    network.train()
    optimizer.zero_grad()
    for idx, (diff_data, phase_amp) in enumerate(loader):
        diff_data, phase_amp = diff_data.to(device), phase_amp.to(device)
        phase_amp_pred = network(diff_data)
        
        loss = network_loss(phase_amp_pred, phase_amp)
        cumu_loss += loss.item()

        loss = loss / accumulation_steps
        loss.backward()

        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            adjusted_loss = cumu_loss / accumulation_steps
            wandb.log({"trn step": epoch * data_size + idx, "trn loss": adjusted_loss})
            epoch_loss += cumu_loss
            cumu_loss = 0
        if (idx + 1) == data_size:
            optimizer.step()
            optimizer.zero_grad()

    if data_size % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return epoch_loss / data_size

def val_epoch(network, loader, data_size, epoch):
  cumu_loss = 0
  network.eval()
  with torch.no_grad():
    for idx, (diff_data, phase_amp) in enumerate(loader):
      diff_data, phase_amp = diff_data.to(device), phase_amp.to(device)
      phase_amp_pred = network(diff_data)
      loss = network_loss(phase_amp_pred, phase_amp)
      cumu_loss += loss.item()
      wandb.log({"val step": epoch*data_size+idx, "val loss": loss.item()})
  return cumu_loss / data_size

def train(config=None):
  with wandb.init(config=config):
    config = wandb.config
    trn_dl, val_dl = build_dataset(config.class_batch)
    val_size = len(val_dl)
    trn_size = len(trn_dl)
    network = build_network()
    optimizer = build_optimizer(network)

    for epoch in range(config.epochs):
        avg_loss = train_epoch(network, trn_dl, optimizer, trn_size, epoch, (batch//config.class_batch))
        wandb.log({"trn loss": avg_loss, "epoch": epoch + 1})
        avg_loss = val_epoch(network, val_dl, val_size, epoch)
        wandb.log({"val loss": avg_loss, "epoch": epoch + 1})
        if(checkpoint and (epoch+1) % checkpoint_rate == 0):
            state = {
                "model": network.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, f'./checkpoint_{epoch+1}.pth')
    del network

sweep_id = wandb.sweep(sweep_config, project="PtychoFormer")
wandb.agent(sweep_id, train, count=num_trails)
