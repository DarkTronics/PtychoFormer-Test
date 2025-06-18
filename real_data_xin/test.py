import os
import wandb
import pprint
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Models.PtychoFormer import PtychoFormer
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_optimizer import Prodigy

import sys
sys.path.append("/home/xguo50/PtychoFormer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# device = "cpu"

os.environ['WANDB_NOTEBOOK_NAME'] = 'train.py'
wandb.login()

num_trails = 20
sweep_config = {'method': 'random'}
metric = {'name': 'trn loss','goal': 'minimize'}
sweep_config['metric'] = metric
parameters_dict = {
    'epochs': {'value': 1000},
    'class_batch': {'value': 1}
  }
sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

batch = 16
checkpoint = True
checkpoint_rate = 20

import torch.nn.functional as F

class Data(Dataset):
    def __init__(self, root_dir, target_resolution=(400, 400)):
        self.file_paths = []
        self.target_resolution = target_resolution
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".pt"):
                    self.file_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index): # currently shaped [25, 2048, 2048]
        data = torch.load(self.file_paths[index])
        diff = data['input']  # Shape: [9, 1700, 1700]
        phase_amp = data['label']  # Shape: [9, 1700, 1700]

        # Add a batch dimension to `diff` and `phase_amp`
        diff = diff.unsqueeze(0)  # Shape: [1, 9, 1700, 1700]
        phase_amp = phase_amp.unsqueeze(0)  # Shape: [1, 9, 1700, 1700]

        # Resize to target resolution
        diff = F.interpolate(diff, size=self.target_resolution, mode='bilinear', align_corners=False)
        phase_amp = F.interpolate(phase_amp, size=self.target_resolution, mode='bilinear', align_corners=False)

        # Remove the batch dimension after resizing
        diff = diff.squeeze(0)  # Shape: [9, target_resolution[0], target_resolution[1]]
        phase_amp = phase_amp.squeeze(0)  # Shape: [9, target_resolution[0], target_resolution[1]]

        return diff, phase_amp


def build_dataset(batch_size):
    # data_dir = '/home/xguo50/PtychoFormer/5by5realdata/ptfiles2'
    data_dir = '/home/xguo50/PtychoFormer/5by5realdata' 
    #'/home/rnakaha2/documents/research/dataset'
    dataset = Data(data_dir)

    # Split dataset into train and validation
    num_data = len(dataset)
    val_size = int(0.1 * num_data)  # 10% for validation
    train_size = num_data - val_size

    train_indices, val_indices = torch.utils.data.random_split(
        range(num_data), [train_size, val_size]
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training Set Size: {len(train_dataset)}")
    print(f"Validation Set Size: {len(val_dataset)}")

    return train_loader, val_loader

def build_network():
  model = PtychoFormer()
  # model.load_state_dict(torch.load('/home/rnakaha2/documents/research/Ptychography/PtychoFormer/Models/Weights/PtychoFormer_weights.pth')['model'])
  # model.load_state_dict(torch.load('/home/xguo50/PtychoFormer/Models/Weights/PtychoFormer_weights.pth')['model'])
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

def get_sample_pred(network, loader):
  network.eval()
  with torch.no_grad():
    diff_data, phase_amp = next(iter(loader))
    diff_data, phase_amp = diff_data.to(device), phase_amp.to(device)
    phase_amp_pred = network(diff_data)

  # Invert amplitude values (reverse colors)
  ampPred = 1.0 - phase_amp_pred[0, 1]  # Invert predicted amplitude
  amp = 1.0 - phase_amp[0, 1]            # Invert ground truth amplitude

  # return phase_amp_pred[0, 0], phase_amp[0, 0], phase_amp_pred[0, 1], phase_amp[0, 1], diff_data[0,0]
  return phase_amp_pred[0, 0], phase_amp[0, 0], ampPred, amp, diff_data[0,0]

def train(config=None):
  with wandb.init(config=config):
    config = wandb.config
    trn_dl, val_dl = build_dataset(config.class_batch)
    val_size = len(val_dl)
    trn_size = len(trn_dl)
    network = build_network()
    optimizer = build_optimizer(network)

    # Resume from checkpoint if available
    # checkpoint_path = '/home/xguo50/PtychoFormer/Models/Weights/PtychoFormer_weights.pth'
    # checkpoint_path = '/home/xguo50/PtychoFormer/checkpoint5by5start.pth'
    checkpoint_path = '/home/xguo50/PtychoFormer/attempt5by5.pth'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint_state = torch.load(checkpoint_path, map_location=device)
        network.load_state_dict(checkpoint_state["model"])
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        start_epoch = checkpoint_state.get("epoch", 0)

    for epoch in range(start_epoch, config.epochs):
        avg_loss = train_epoch(network, trn_dl, optimizer, trn_size, epoch, (batch//config.class_batch))
        wandb.log({"trn loss": avg_loss, "epoch": epoch + 1})
        avg_loss = val_epoch(network, val_dl, val_size, epoch)
        wandb.log({"val loss": avg_loss, "epoch": epoch + 1})
        phase_pred, phase, amp_pred, amp, diff = get_sample_pred(network, val_dl)
        wandb.log({"val diff": wandb.Image(diff),"val ph pred": wandb.Image(phase_pred), "val amp pred": wandb.Image(amp_pred), "val ph truth": wandb.Image(phase), "val amp truth": wandb.Image(amp)})  
        phase_pred, phase, amp_pred, amp, diff = get_sample_pred(network, trn_dl)
        wandb.log({"trn diff": wandb.Image(diff),"trn ph pred": wandb.Image(phase_pred), "trn amp pred": wandb.Image(amp_pred), "trn ph truth": wandb.Image(phase), "trn amp truth": wandb.Image(amp)})  

        if(checkpoint and (epoch+1) % checkpoint_rate == 0):
            state = {
                "model": network.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, f'./checkpoint5by5.pth')
    state = {
        "model": network.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, f'./attempt5by5.pth')
    del network

sweep_id = wandb.sweep(sweep_config, project="PtychoFormer")
wandb.agent(sweep_id, train, count=num_trails)
