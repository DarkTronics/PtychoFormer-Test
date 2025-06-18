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

os.environ['WANDB_NOTEBOOK_NAME'] = 'train.py'
wandb.login()
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'trn loss',
        'goal': 'minimize'
    },
    'parameters': {
        'epochs': {'value': 1000},
        'class_batch': {'value': 1}
    }
}
num_trials = 5
pprint.pprint(sweep_config)

batch = 16
checkpoint = True
checkpoint_rate = 20

class Data(Dataset):
    def __init__(self, root_dir, target_resolution=(400, 400)):
        self.file_paths = [
            os.path.join(subdir, file)
            for subdir, _, files in os.walk(root_dir)
            for file in files if file.endswith(".pt")
        ]
        self.target_resolution = target_resolution

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        data = torch.load(self.file_paths[index]) # currently shaped [25, 2048, 2048]
        diff = data['input']  # Shape: [25, 2048, 2048]
        phase_amp = data['label']  # Shape: [2, 2048, 2048]

        # Resize tensors (F.interpolate requires 4D input)
        diff = F.interpolate(
            diff.unsqueeze(0),  # Shape: [1, 25, 2048, 2048]
            size=self.target_resolution,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Shape: [25, 2048, 2048]

        phase_amp = F.interpolate(
            phase_amp.unsqueeze(0),  # Shape: [1, 2, 2048, 2048]
            size=self.target_resolution,
            mode='bilinear',  # Consider 'nearest' if labels are categorical
            align_corners=False
        ).squeeze(0)  # Shape: [25, 2048, 2048]

        return diff, phase_amp

def build_dataset_two_files(batch_size):
    # data_dir = '/home/xguo50/PtychoFormer/5by5realdata/twofile'  # Folder with two .pt files
    # data_dir = '/home/xguo50/PtychoFormer/5by5realdata/testdata'
    data_dir = '/home/xguo50/PtychoFormer/5by5realdata/testdata'
    dataset = Data(data_dir)
    
    # Ensure we have exactly two files
    if len(dataset) != 2:
        raise ValueError(f"Expected exactly 2 files, but found {len(dataset)} files.")
    
    # Use file 0 for training and file 1 for validation
    train_dataset = Subset(dataset, [0])
    val_dataset = Subset(dataset, [1])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    print(f"Training using file: {dataset.file_paths[0]}")
    print(f"Validation using file: {dataset.file_paths[1]}")
    
    return train_loader, val_loader
    
def build_dataset_for_one(batch_size):
    data_dir = '/home/xguo50/PtychoFormer/5by5realdata/onefile'  # Folder with one .pt file
    dataset = Data(data_dir)

    if len(dataset) < 2:
        # No validation split for 1 sample
        print("Single image dataset detected.")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        return train_loader, val_loader

def build_dataset(batch_size):
    data_dir = '/home/xguo50/PtychoFormer/1440realdata2'
    # data_dir = '/home/xguo50/PtychoFormer/1440realdata2/03'
    dataset = Data(data_dir)
    generator = torch.Generator().manual_seed(42)

    # Split into train and validation
    num_data = len(dataset)
    val_size = int(0.1 * num_data)
    train_size = num_data - val_size
    # train_indices, val_indices = torch.utils.data.random_split(
    #     range(num_data), [train_size, val_size]
    # )
    train_indices, val_indices = torch.utils.data.random_split(
        range(num_data), [train_size, val_size], generator=generator
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # DataLoaders with parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  
        pin_memory=True  
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader

def build_network(checkpoint_path=None):
    model = PtychoFormer()
    model = model.to(device)

    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    return model

def build_optimizer(network):
  optimizer = Prodigy(network.parameters(), lr=1)
  return optimizer

def network_loss(pred, target, norm = True):
    l1_fn = nn.L1Loss()
    if norm:
      pred[:, 0:1, :, :] = (pred[:, 0:1, :, :] + np.pi) / (2 * np.pi)
      target[:, 0:1, :, :]  = (target[:, 0:1, :, :] + np.pi) / (2 * np.pi)
    return l1_fn(pred, target)

def train_epoch(network, loader, optimizer, data_size, epoch, accumulation_steps):
    if data_size == 1 or data_size == 2:
       accumulation_steps = 1
    epoch_loss = 0
    cumu_loss = 0
    network.train()
    optimizer.zero_grad()
    for idx, (diff_data, phase_amp) in enumerate(loader):
        diff_data, phase_amp = diff_data.to(device), phase_amp.to(device)
        phase_amp_pred = network(diff_data)

        loss = network_loss(phase_amp_pred, phase_amp)
        wandb.log({"trn step": epoch * data_size + idx, "trn loss": loss.item()})
        cumu_loss += loss.item()

        loss = loss / accumulation_steps
        loss.backward()

        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            adjusted_loss = cumu_loss / accumulation_steps
            # wandb.log({"trn step": epoch * data_size + idx, "trn loss": adjusted_loss})
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

# def get_sample_pred(network, loader, index=0):
#   network.eval()
#   with torch.no_grad():
#     diff_data, phase_amp = next(iter(loader))
#     diff_data, phase_amp = diff_data.to(device), phase_amp.to(device)
#     phase_amp_pred = network(diff_data)

#   # Invert amplitude values (reverse colors)
#   ampPred = 1.0 - phase_amp_pred[0, 1]  # Invert predicted amplitude
#   amp = 1.0 - phase_amp[0, 1]            # Invert ground truth amplitude

#   # return phase_amp_pred[0, 0], phase_amp[0, 0], phase_amp_pred[0, 1], phase_amp[0, 1], diff_data[0,0]
#   return phase_amp_pred[0, 0], phase_amp[0, 0], ampPred, amp, diff_data[0,0]

def get_sample_pred(network, dataset, index=0):
    network.eval()
    with torch.no_grad():
        sample = dataset[index]  # Always pick the same data point
        diff_data, phase_amp = sample
        diff_data = diff_data.unsqueeze(0).to(device)
        phase_amp = phase_amp.unsqueeze(0).to(device)

        # Forward pass
        phase_amp_pred = network(diff_data)

    # Invert amplitude values
    # amp_pred = 1.0 - phase_amp_pred[0, 1]
    # amp = 1.0 - phase_amp[0, 1]
    return phase_amp_pred[0, 0], phase_amp[0, 0], 1-phase_amp_pred[0, 1], 1-phase_amp[0, 1], diff_data[0, 0] # amp_pred, amp, phase_pred, phase, diff

def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

def normalize_for_wandb(tensor):
    """Normalize a 2D or 3D tensor to [0, 255] and convert to uint8 numpy."""
    tensor = tensor.detach().cpu().float()
    arr = tensor.numpy()

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # remove singleton channel

    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()

    arr = (arr * 255).astype(np.uint8)
    return arr

def train_epoch_no_accumulation(network, loader, optimizer, epoch):
    network.train()
    epoch_loss = 0

    for idx, (diff_data, phase_amp) in enumerate(loader):
        diff_data, phase_amp = diff_data.to(device), phase_amp.to(device)
        phase_amp_pred = network(diff_data)

        loss = network_loss(phase_amp_pred, phase_amp)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"trn step": epoch * len(loader) + idx, "trn loss": loss.item()})
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def val_epoch_no_accumulation(network, loader, epoch):
    network.eval()
    cumu_loss = 0
    with torch.no_grad():
        for idx, (diff_data, phase_amp) in enumerate(loader):
            diff_data, phase_amp = diff_data.to(device), phase_amp.to(device)
            phase_amp_pred = network(diff_data)
            loss = network_loss(phase_amp_pred, phase_amp)
            cumu_loss += loss.item()
            wandb.log({"val step": epoch * len(loader) + idx, "val loss": loss.item()})
    return cumu_loss / len(loader)

def train(config=None):
  with wandb.init(config=config):
    config = wandb.config

    # change this build dataset for testing one, two, or full dataset
    # trn_dl, val_dl = build_dataset(config.class_batch)
    trn_dl, val_dl = build_dataset(config.class_batch)
    val_size = len(val_dl)
    trn_size = len(trn_dl)
    print(f"Training Set Size: {trn_size}")
    print(f"Validation Set Size: {val_size}")

    # network = build_network()
    # optimizer = build_optimizer(network)
    # optimizer = Prodigy(network.parameters(), lr=1e-4)

    # pretrained_path = "/home/xguo50/PtychoFormer/myWeights/try1using5by5badval/attempt5by5.pth"
    pretrained_path = "/home/xguo50/PtychoFormer/myWeights/5by5checkpoint_13.pth" # start here
    # resume_path = "/home/xguo50/PtychoFormer/attempt.pth"  # your last saved weights
    resume_path = "/home/xguo50/PtychoFormer/attempt.pth"  # your last saved weights

    if os.path.exists(resume_path):
        network = build_network(checkpoint_path=resume_path)
        print(f"Resuming from {resume_path}")
    else:
        network = build_network(checkpoint_path=pretrained_path)
        print(f"Loading pretrained weights from {pretrained_path}")

    # optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    optimizer = build_optimizer(network)

    for epoch in range(config.epochs):
        wandb.log({"epoch": epoch + 1})
        avg_loss = train_epoch(network, trn_dl, optimizer, trn_size, epoch, (batch//config.class_batch))
        avg_loss = val_epoch(network, val_dl, val_size, epoch)
        # avg_loss = train_epoch_no_accumulation(network, trn_dl, optimizer, epoch)
        # avg_val_loss = val_epoch_no_accumulation(network, val_dl, epoch)
        for i in range(5):
            amp_pred, amp, phase_pred, phase, diff = get_sample_pred(network, val_dl.dataset, index=i)
            wandb.log({
                # f"val diff {i}": wandb.Image(normalize_for_wandb(diff)),
                f"val ph pred {i}": wandb.Image(normalize_for_wandb(phase_pred)),
                f"val amp pred {i}": wandb.Image(normalize_for_wandb(amp_pred)),
                f"val ph truth {i}": wandb.Image(normalize_for_wandb(phase)),
                f"val amp truth {i}": wandb.Image(normalize_for_wandb(amp))
            })

            # Training samples
            amp_pred, amp, phase_pred, phase, diff = get_sample_pred(network, trn_dl.dataset, index=i)
            wandb.log({
                # f"trn diff {i}": wandb.Image(normalize_for_wandb(diff)),
                f"trn ph pred {i}": wandb.Image(normalize_for_wandb(phase_pred)),
                f"trn amp pred {i}": wandb.Image(normalize_for_wandb(amp_pred)),
                f"trn ph truth {i}": wandb.Image(normalize_for_wandb(phase)),
                f"trn amp truth {i}": wandb.Image(normalize_for_wandb(amp))
            })

        if(checkpoint and (epoch+1) % checkpoint_rate == 0):
            state = {
                "model": network.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, f'./checkpoint.pth')
    state = {
        "model": network.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, f'./attempt.pth')
    del network

sweep_id = wandb.sweep(sweep_config, project="PtychoFormerXin")
wandb.agent(sweep_id, train, count=num_trials)
