from Models.PtychoFormer import PtychoFormer
import torch
import torch.nn as nn
from typing import List
import numpy as np
import time
import sys
import math
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from torch.fft import ifft2, fft2, ifftshift, fftshift


class ExtendedPtychoFormer(nn.Module):
    def __init__(self, in_channels: int = 9, widths: List[int] = [64, 128, 256, 512], 
                 depths: List[int] = [3, 4, 6, 3], all_num_heads: List[int] = [1, 2, 4, 8],
                 patch_sizes: List[int] = [5, 3, 3, 3], overlap_sizes: List[int] = [2, 2, 2, 2],
                 reduction_ratios: List[int] = [8, 4, 2, 1], mlp_expansions: List[int] = [4, 4, 4, 4],
                 decoder_channels: int = 256, drop_prob: float = 0.1):
        super().__init__()
        
        # Initialize PtychoFormer internally
        self.PtychoFormer = PtychoFormer(
            in_channels, widths, depths, all_num_heads, patch_sizes, overlap_sizes, reduction_ratios, mlp_expansions, decoder_channels, drop_prob
        )
        self.PtychoFormer.eval()
    def create_probe(self, r1=1):
        pixelnum = 128
        lam = torch.tensor(632.8 * 10**(-9), dtype=torch.float64)
        pixelsize = torch.tensor(7.4 * 10**(-6), dtype=torch.float64)
        d1 = torch.tensor(10 * 10**(-3), dtype=torch.float64)
        d2 = torch.tensor(20 * 10**(-3), dtype=torch.float64)
        RI = torch.tensor(r1, dtype=torch.float64)
        Frequency = 1 / pixelsize
        
        Fxvector = torch.linspace(-Frequency / 2, Frequency / 2, pixelnum, dtype=torch.float64)
        Fyvector = torch.linspace(-Frequency / 2, Frequency / 2, pixelnum, dtype=torch.float64)
        [FxMat, FyMat] = torch.meshgrid(Fxvector, Fyvector, indexing='ij')
        
        xx = torch.linspace(-pixelsize * pixelnum / 2, pixelsize * pixelnum / 2, pixelnum, dtype=torch.float64)
        yy = torch.linspace(-pixelsize * pixelnum / 2, pixelsize * pixelnum / 2, pixelnum, dtype=torch.float64)
        [XX, YY] = torch.meshgrid(xx, yy, indexing='ij')
        
        Hole = (torch.sqrt(XX**2 + YY**2) <= 3 * 10**(-4))
        
        Hz1 = torch.exp((1j * 2 * np.pi * d1 * RI / lam) * torch.sqrt(1 - (lam * FxMat / RI)**2 - (lam * FyMat / RI)**2))
        Hz2 = torch.exp((1j * 2 * np.pi * d2 * RI / lam) * torch.sqrt(1 - (lam * FxMat / RI)**2 - (lam * FyMat / RI)**2))
        HzMinus = torch.exp((-1j * 2 * np.pi * d2 * RI / lam) * torch.sqrt(1 - (lam * FxMat / RI)**2 - (lam * FyMat / RI)**2))
        
        amplitude1 = torch.ones((pixelnum, pixelnum), dtype=torch.float64)
        phase1 = torch.zeros((pixelnum, pixelnum), dtype=torch.float64)
        wave1 = amplitude1 * torch.exp(1j * phase1)
        
        Wave1 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Hole))) * wave1)))
        Afield = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Wave1))) * Hz1)))
        
        return Afield, Hole, Hz2, HzMinus
    
    def load_ptychoformer_weights(self, pretrained_weights):
        """Loads weights into the internal PtychoFormer model"""
        self.PtychoFormer.load_state_dict(pretrained_weights)

    def feathered_stitching(self, data, n_grid, step, diff_step=2):
        '''
        Stitching algorithm for 3x3 grid prediction using feathering.

        Inputs:
        - data: local prediction in n x n grid
        - n_grid: square root of number of grid (n)
        - step: pixel-wise lateral offset for each scanned position 
        - diff_step: Step size (in terms of number of diffraction patterns) for each adjacent 3x3 grid window  
        '''

        device = data.device
        N, M, _, patch_size, _ = data.shape
        diff_size = 128
        offset = 36
        delta = 0
        patch_size = patch_size-2*offset

        if(diff_step == 1):
            delta = step
        if(diff_step == 3):
            delta = -step
        overlap = delta + diff_size - 2*offset
        patch_step = patch_size - overlap
        recon = torch.zeros((2, patch_step*(n_grid-1) + patch_size + 2*offset, patch_step*(n_grid-1) + patch_size + 2*offset), device=device)

        for n in range(N):
            for m in range(M):
                phase = data[n,m,0]
                amp = data[n,m,1]
                if(overlap > 0):
                    if not (n == 0):
                        gradient = torch.linspace(0, 1, steps=overlap, device=device).unsqueeze(1)
                        phase = phase[offset:, :]
                        amp = amp[offset:, :]
                        phase[0:overlap, :] *= gradient
                        amp[0:overlap, :] *= gradient
                    if not (m == 0):
                        gradient = torch.linspace(0, 1, steps=overlap, device=device)
                        phase = phase[:, offset:]
                        amp = amp[:, offset:]
                        phase[:, 0:overlap] *= gradient
                        amp[:, 0:overlap] *= gradient
                    if not (n == N-1):
                        gradient = torch.linspace(1, 0, steps=overlap, device=device).unsqueeze(1)
                        phase = phase[:-offset, :]
                        amp = amp[:-offset, :]
                        phase[-overlap:, :] *= gradient
                        amp[-overlap:, :] *= gradient
                    if not (m == M-1):
                        gradient = torch.linspace(1, 0, steps=overlap, device=device)
                        phase = phase[:, :-offset]
                        amp = amp[:, :-offset]
                        phase[:, -overlap:] *= gradient
                        amp[:, -overlap:] *= gradient
                if not (n == 0) and not (m == 0) and not (n == N-1) and not (m == M-1):
                    recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ offset] = torch.where(
                        recon[0, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ offset] == 0, # condition
                        phase, 
                        torch.where(phase == 0, 
                            recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ offset],
                            (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ offset])))
                    recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ offset] = torch.where(
                        recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ offset] == 0, # condition
                        amp, 
                        torch.where(amp == 0, 
                            recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ offset],
                            (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ offset])))
                if not (n == 0) and not (m == 0) and not (n == N-1) and (m == M-1):
                    recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                        recon[0, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                        phase, 
                        torch.where(phase == 0, 
                            recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                            (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])))
                    recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                        recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                        amp, 
                        torch.where(amp == 0, 
                            recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                            (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])))
                if not (n == 0) and not (m == 0) and (n == N-1) and not (m == M-1):
                    recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset] = torch.where(
                        recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset] == 0, # condition
                        phase, 
                        torch.where(phase == 0, 
                            recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset],
                            (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset])))
                    recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset] = torch.where(
                        recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset] == 0, # condition
                        amp, 
                        torch.where(amp == 0, 
                            recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset],
                            (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset])))
                if (n == 0) and not (m == 0) and not (n == N-1) and not (m == M-1):
                    recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset] = torch.where(
                        recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset] == 0, # condition
                        phase, 
                        torch.where(phase == 0, 
                            recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset],
                            (phase + recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset])))
                    recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset] = torch.where(
                        recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset] == 0, # condition
                        amp, 
                        torch.where(amp == 0, 
                            recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset],
                            (amp + recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset])))
                if not (n == 0) and (m == 0) and not (n == N-1) and not (m == M-1):
                    recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] = torch.where(
                        recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] == 0, # condition
                        phase, 
                        torch.where(phase == 0, 
                            recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset],
                            (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset])))
                    recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] = torch.where(
                        recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] == 0, # condition
                        amp, 
                        torch.where(amp == 0, 
                            recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset],
                            (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset])))
                if (n == 0) and (m == 0) and not (n == N-1) and not (m == M-1):
                    recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] = torch.where(
                        recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] == 0, # condition
                        phase, 
                        torch.where(phase == 0, 
                            recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset],
                            (phase + recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset])))
                    recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] = torch.where(
                        recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] == 0, # condition
                        amp, 
                        torch.where(amp == 0, 
                            recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset],
                            (amp + recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset])))
                if not (n == 0) and not (m == 0) and (n == N-1) and (m == M-1):
                    recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                        recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                        phase, 
                        torch.where(phase == 0, 
                            recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                            (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])))
                    recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                        recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                        amp, 
                        torch.where(amp == 0, 
                            recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                            (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])))
                if (n == 0) and not (m == 0) and not (n == N-1) and (m == M-1):
                    recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                        recon[0, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                        phase, 
                        torch.where(phase == 0, 
                            recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                            (phase + recon[0, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])))
                    recon[1, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                        recon[1, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                        amp, 
                        torch.where(amp == 0, 
                            recon[1, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                            (amp + recon[1, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])))
                if not (n == 0) and (m == 0) and (n == N-1) and not (m == M-1):
                    recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step:m*patch_step+patch_size+ offset] = torch.where(
                        recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset] == 0, # condition
                        phase, 
                        torch.where(phase == 0, 
                            recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step:m*patch_step+patch_size+ offset],
                            (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset])))
                    recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset] = torch.where(
                        recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset] == 0, # condition
                        amp, 
                        torch.where(amp == 0, 
                            recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset],
                            (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset])))
        return recon
    def PtychoFormer_initialization(self, PtychoFormer_pred):
        return PtychoFormer_pred[1]*torch.exp(1j*PtychoFormer_pred[0]).to(torch.complex128)
    
    def ePIE(self, diffraction, step, iteration, Ref, n_row, probe_upd, alpha, beta, record, verbose, plot_freq):
        '''
        This function runs the extended-ptychographic iterative engine (ePIE). This implementation only allows for grid scans.

        Inputs:
        - diffraction: input diffraction patterns in n x n grid
        - step: lateral offsets between each scan position
        - iteration: number of iteration for ePIE
        - n_row: square root of the number of diffraction patterns (n)
        - Ref: initial tranmission estimate
        - probe_upd: if true, update probe function, else don't update probe function  
        - record: if True, record the sum squared error per iteration
        - verbose: print the current iteration count and plot the current estimated transmission and probe function
        - plot_freq: the frequency of plot generation
        - alpha: update speed of estimated transmission function
        - beta: update speed of estimated probe function
        '''
        device = diffraction.device
        a = 0.05
        b = 0.05
        pixelnum = 128
        Wave, Hole, Hz2, HzMinus = self.create_probe()
        Wave, Hole, Hz2, HzMinus = Wave.to(device), Hole.to(device), Hz2.to(device), HzMinus.to(device)
        hist_loss = []
        new = torch.zeros((pixelnum,pixelnum)).to(torch.complex128).to(device)
        start_time = time.time()
        with torch.no_grad():
            for iter in range(iteration):
                if(verbose):
                    sys.stdout.write("\rIteration:"+ str(iter+1))
                    sys.stdout.flush()
                for m in range(n_row):
                    for n in range(n_row):
                        amplitude = torch.sqrt(diffraction[m * n_row + n])

                        new = Ref[m * step : m * step + pixelnum, n * step : n * step + pixelnum] * Hole.to(torch.complex128)
                        WaveRef = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(new * Wave))) * Hz2)))

                        WaveRef1 = amplitude*torch.exp(1j*torch.angle(WaveRef)).to(torch.complex128)
                        WaveRefr1 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(WaveRef1)))*HzMinus)))
                        
                        new = new + alpha*((abs(Wave) * torch.conj(Wave)) / (torch.max(abs(Wave)) * (abs(Wave)**2 + a))) * (WaveRefr1 - Wave * new)
                        Ref[m * step : m * step + pixelnum, n * step : n * step + pixelnum][Hole == 1] = new[Hole == 1]
                        if probe_upd:
                            Wave = Wave + beta*((abs(new) * torch.conj(new)) / (torch.max(abs(new) * pixelnum) * (abs(new)**2 + b)))*(WaveRefr1 - Wave * new)

                if(record):
                    # compute SSE loss
                    SSELoss = 0
                    for m in range(n_row):
                        for n in range(n_row):
                            truth = torch.sqrt(diffraction[m * n_row + n])
                            new = Ref[m * step : m * step + pixelnum, n * step : n * step + pixelnum] * Hole
                            guess = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(new * Wave))) * Hz2)))
                            guess = abs(guess)
                            SSELoss += nn.functional.mse_loss(guess, truth).item()
                    hist_loss.append(SSELoss)
                if(verbose and ((iter+1) % plot_freq == 0)):
                    f, ax = plt.subplots(1,4, figsize=(15,12))
                    amp = abs(Ref).detach().cpu()
                    pha = torch.angle(Ref).detach().cpu()
                    ax[0].imshow(pha[0:(n_row-1)*step + pixelnum, 0:(n_row-1)*step + pixelnum], cmap='gray')
                    ax[1].imshow(amp[0:(n_row-1)*step + pixelnum, 0:(n_row-1)*step + pixelnum], cmap='gray')
                    ax[2].imshow(abs(Wave).detach().cpu(), cmap='gray')
                    ax[3].imshow(torch.angle(Wave).detach().cpu(), cmap='gray')
                    plt.show(block=False)
            total_time = (time.time() - start_time)
            Amplitude = abs(Ref)
            Phase = torch.tensor(unwrap_phase(torch.angle(Ref).cpu())).to(device)
            final_pred = torch.stack((Phase, Amplitude), axis=0)
        return final_pred, hist_loss, total_time

    def forward(self, grid_data, single_data, step, diff_step, n_row, iteration=1000, probe_upd = True, alpha=1, beta=1, record = False, verbose = False, plot_freq = 20):
        # Get PtychoFormer prediction
        pred = self.PtychoFormer(grid_data)
        # Stitch together local reconstructions
        n_grid = int(math.sqrt(pred.shape[0]))
        patch_size = pred.shape[-1]
        stitched_pred = self.feathered_stitching(pred.reshape(n_grid, n_grid, 2, patch_size, patch_size), n_grid, step, diff_step)
        # Initial prediction
        Ref_init = self.PtychoFormer_initialization(stitched_pred)
        # Call ePIE for iterative refinement
        final_pred, hist_loss, total_time = self.ePIE(single_data, step, iteration, Ref_init, n_row, probe_upd, alpha, beta, record, verbose, plot_freq)
        return final_pred, hist_loss, total_time
