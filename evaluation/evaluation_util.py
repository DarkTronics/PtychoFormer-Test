import torch
from torch.fft import ifft2, fft2, ifftshift, fftshift
import numpy as np
import matplotlib.pyplot as plt

def create_probe(r1=1):
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

def generate_grids(transmission, step, n_grid, diff_step=2):
    Afield, Hole, Hz2, _ = create_probe()
    n_diff = 3
    diff_size = 128
    patch_size = diff_size + (n_diff - 1) * step
    max_row = n_diff + (n_grid - 1) * diff_step
    max_pixel = diff_size + (max_row - 1) * step

    # Create label masking per patch
    mask = torch.zeros((patch_size, patch_size), dtype=torch.float64)
    for n in range(n_diff):
        for m in range(n_diff):
            mask[n*step : n*step+diff_size, m*step : m*step+diff_size] = torch.where(
                mask[n*step : n*step+diff_size, m*step : m*step+diff_size] == 0,
                Hole,
                mask[n*step : n*step+diff_size, m*step : m*step+diff_size]
            )

    # Create label masking for entire image
    entire_mask = torch.zeros((max_pixel, max_pixel), dtype=torch.float64)
    for n in range(max_row):
        for m in range(max_row):
            entire_mask[n*step : n*step+diff_size, m*step : m*step+diff_size] = torch.where(
                entire_mask[n*step : n*step+diff_size, m*step : m*step+diff_size] == 0,
                Hole,
                entire_mask[n*step : n*step+diff_size, m*step : m*step+diff_size]
            )

    ph = torch.tensor(transmission[0][0:max_pixel, 0:max_pixel], dtype=torch.float64)
    amp = torch.tensor(transmission[1][0:max_pixel, 0:max_pixel], dtype=torch.float64)
    
    # ph = (ph - ph.min()) / (ph.max() - ph.min())
    # ph = np.pi * (2 * ph - 1)
    # amp = (amp - amp.min()) / (amp.max() - amp.min())
    # amp = amp * (1 - 0.05) + 0.05

    # Create complex amp
    C = amp * torch.exp(1j * ph)
    
    grid_data = torch.zeros((n_grid * n_grid, n_diff * n_diff, patch_size, patch_size), dtype=torch.float64)
    single_data = torch.zeros((max_row * max_row, diff_size, diff_size), dtype=torch.float64)
    patch_label = torch.zeros((n_grid * n_grid, 2, patch_size, patch_size), dtype=torch.float64)

    # Generate 3x3 grid data for PtychoFormer
    for i in range(n_grid):
        for j in range(n_grid):
            patch_label[i * n_grid + j, 0] = mask * ph[i * (diff_step * step) : i * (diff_step * step) + patch_size, j * (diff_step * step) : j * (diff_step * step) + patch_size]
            patch_label[i * n_grid + j, 1] = mask * amp[i * (diff_step * step) : i * (diff_step * step) + patch_size, j * (diff_step * step) : j * (diff_step * step) + patch_size]
            for n in range(n_diff):
                for m in range(n_diff):
                    Temp = C[
                        i * (diff_step * step) + n * step : i * (diff_step * step) + n * step + diff_size,
                        j * (diff_step * step) + m * step : j * (diff_step * step) + m * step + diff_size
                    ] * Hole
                    Assumption2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Afield * Temp))) * Hz2)))
                    grid_data[i * n_grid + j, n * n_diff + m, n * step : n * step + diff_size, m * step : m * step + diff_size] = abs(Assumption2)**2
    # Generate individual diffraction data for ePIE and CNN models
    for i in range(max_row):
        for j in range(max_row):
            Temp = C[i * step : i * step + diff_size, j * step : j * step + diff_size] * Hole
            Assumption2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Afield * Temp))) * Hz2)))
            single_data[i * max_row + j] = abs(Assumption2)**2
    
    entire_label = torch.stack((ph * entire_mask, amp * entire_mask), axis=0)
    return grid_data.float(), single_data.float(), patch_label, entire_label


def feathered_stitching(data, n_grid, step, diff_step=2):
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

def unfeathered_stitching(data, n_grid, step, diff_step=2):
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
            phase = data[n,m,0].clone()
            amp = data[n,m,1].clone()
            if(overlap > 0):
                if not (n == 0):
                    phase = phase[offset:, :]
                    amp = amp[offset:, :]
                if not (m == 0):
                    phase = phase[:, offset:]
                    amp = amp[:, offset:]
                if not (n == N-1):
                    phase = phase[:-offset, :]
                    amp = amp[:-offset, :]
                if not (m == M-1):
                    phase = phase[:, :-offset]
                    amp = amp[:, :-offset]
            if not (n == 0) and not (m == 0) and not (n == N-1) and not (m == M-1):
                recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ offset] = torch.where(
                    recon[0, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ offset] == 0, # condition
                    phase, 
                    torch.where(phase == 0, 
                        recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ offset],
                        (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ offset])/2))
                recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ offset] = torch.where(
                    recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ offset] == 0, # condition
                    amp, 
                    torch.where(amp == 0, 
                        recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ offset],
                        (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ offset])/2))
            if not (n == 0) and not (m == 0) and not (n == N-1) and (m == M-1):
                recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                    recon[0, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                    phase, 
                    torch.where(phase == 0, 
                        recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                        (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])/2))
                recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                    recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                    amp, 
                    torch.where(amp == 0, 
                        recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                        (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])/2))
            if not (n == 0) and not (m == 0) and (n == N-1) and not (m == M-1):
                recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset] = torch.where(
                    recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset] == 0, # condition
                    phase, 
                    torch.where(phase == 0, 
                        recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset],
                        (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset])/2))
                recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset] = torch.where(
                    recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset] == 0, # condition
                    amp, 
                    torch.where(amp == 0, 
                        recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset],
                        (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ offset])/2))
            if (n == 0) and not (m == 0) and not (n == N-1) and not (m == M-1):
                recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset] = torch.where(
                    recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset] == 0, # condition
                    phase, 
                    torch.where(phase == 0, 
                        recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset],
                        (phase + recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset])/2))
                recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset] = torch.where(
                    recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset] == 0, # condition
                    amp, 
                    torch.where(amp == 0, 
                        recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset],
                        (amp + recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size + offset])/2))
            if not (n == 0) and (m == 0) and not (n == N-1) and not (m == M-1):
                recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] = torch.where(
                    recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] == 0, # condition
                    phase, 
                    torch.where(phase == 0, 
                        recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset],
                        (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset])/2))
                recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] = torch.where(
                    recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] == 0, # condition
                    amp, 
                    torch.where(amp == 0, 
                        recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset],
                        (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset])/2))
            if (n == 0) and (m == 0) and not (n == N-1) and not (m == M-1):
                recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] = torch.where(
                    recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] == 0, # condition
                    phase, 
                    torch.where(phase == 0, 
                        recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset],
                        (phase + recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset])/2))
                recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] = torch.where(
                    recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset] == 0, # condition
                    amp, 
                    torch.where(amp == 0, 
                        recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset],
                        (amp + recon[1, n*patch_step:n*patch_step + patch_size + offset, m*patch_step:m*patch_step+patch_size + offset])/2))
            if not (n == 0) and not (m == 0) and (n == N-1) and (m == M-1):
                recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                    recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                    phase, 
                    torch.where(phase == 0, 
                        recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                        (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])/2))
                recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                    recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                    amp, 
                    torch.where(amp == 0, 
                        recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                        (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])/2))
            if (n == 0) and not (m == 0) and not (n == N-1) and (m == M-1):
                recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                    recon[0, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                    phase, 
                    torch.where(phase == 0, 
                        recon[0, n*patch_step:n*patch_step + patch_size + offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                        (phase + recon[0, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])/2))
                recon[1, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] = torch.where(
                    recon[1, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset] == 0, # condition
                    amp, 
                    torch.where(amp == 0, 
                        recon[1, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset],
                        (amp + recon[1, n*patch_step:n*patch_step + patch_size+ offset, m*patch_step + offset:m*patch_step+patch_size+ 2*offset])/2))
            if not (n == 0) and (m == 0) and (n == N-1) and not (m == M-1):
                recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step:m*patch_step+patch_size+ offset] = torch.where(
                    recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset] == 0, # condition
                    phase, 
                    torch.where(phase == 0, 
                        recon[0, n*patch_step + offset:n*patch_step + patch_size + 2*offset, m*patch_step:m*patch_step+patch_size+ offset],
                        (phase + recon[0, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset])/2))
                recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset] = torch.where(
                    recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset] == 0, # condition
                    amp, 
                    torch.where(amp == 0, 
                        recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset],
                        (amp + recon[1, n*patch_step + offset:n*patch_step + patch_size+ 2*offset, m*patch_step:m*patch_step+patch_size+ offset])/2))
    return recon

def CNN_stitching(data, n_grid, step):
    device = data.device
    N, M, _, patch_size, _ = data.shape

    # overlap = patch_size - step
    recon = torch.zeros((2, step*(n_grid-1) + patch_size, step*(n_grid-1) + patch_size), device=device)
    for n in range(N):
        for m in range(M):
            phase = data[n,m,0]
            amp = data[n,m,1]
            recon[0, n*step:n*step + patch_size, m*step:m*step+patch_size] = torch.where(
                recon[0, n*step:n*step + patch_size, m*step:m*step+patch_size] == 0, 
                phase, 
                (phase + recon[0, n*step:n*step + patch_size, m*step:m*step+patch_size])/2)
            recon[1, n*step:n*step + patch_size, m*step:m*step+patch_size] = torch.where(
                recon[1, n*step:n*step + patch_size, m*step:m*step+patch_size] == 0,
                amp, 
                (amp + recon[1, n*step:n*step + patch_size, m*step:m*step+patch_size])/2)
    return recon

def compute_nrmse(reconstructed_image, ground_truth_image, diff_bool=True):
    if diff_bool:
        diff = reconstructed_image - ground_truth_image
        diff = torch.mean(diff)
        # Compute the difference with the optimal phase shift applied
        difference = reconstructed_image - diff - ground_truth_image
    else:
        difference = reconstructed_image - ground_truth_image
    # Calculate the NRMSE
    nrmse = np.linalg.norm(difference) / np.linalg.norm(ground_truth_image)
    
    return nrmse


def create_ring_mask(rows, cols, center_row, center_col, radius, width):
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
    mask = (distance >= radius) & (distance < radius + width)
    return mask
def compute_frc(image1, image2, ring_width=1):
    # Compute the Fourier transforms
    F1 = fftshift(fft2(image1))
    F2 = fftshift(fft2(image2))
    
    # Get the shape and center of the images
    rows, cols = F1.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Compute the maximum radius
    max_radius = int(np.hypot(center_row, center_col))
    
    frc_values = []
    radii = np.arange(0, max_radius, ring_width)
    
    for radius in radii:
        # Create a mask for the current ring
        mask = create_ring_mask(rows, cols, center_row, center_col, radius, ring_width)
        
        # Extract the values in the current ring for both images
        F1_ring = np.array(F1[mask])
        F2_ring = np.array(F2[mask])
        
        # Compute the correlation
        num = np.sum(F1_ring * np.conj(F2_ring))
        den = np.sqrt(np.sum(np.abs(F1_ring) ** 2) * np.sum(np.abs(F2_ring) ** 2))
        frc_value = np.abs(num) / den
        frc_values.append(frc_value)
    
    # Normalize the spatial frequency to range from 0 to 1
    nyquist_frequency = max_radius
    normalized_frequencies = radii / nyquist_frequency
    
    return normalized_frequencies, frc_values

def extend_to_one(frequencies, data_dict):
    # Check if the last frequency is 1.0
    if frequencies[-1] < 1.0:
        # Append 1.0 to the frequencies
        extended_frequencies = np.append(frequencies, 1.0)
        
        # Repeat the last value of each measurement until frequency 1.0
        for key in data_dict:
            data_dict[key] = np.append(data_dict[key], data_dict[key][-1])
    else:
        extended_frequencies = frequencies
    
    return extended_frequencies, data_dict

def plot_frc_multiple(frequencies, frc_data, threshold=0.143, stdev_step=5):
    fig, (ax_amp, ax_ph) = plt.subplots(1, 2, figsize=(14, 5))  # Create two subplots

    labels = ['PtychoFormer Amp', 'PtychoFormer', 'ePF Amp', 'ePF', 'ePIE Amp', 'ePIE', ]
    phase_colors = ['#1f77b4', '#38b000', '#ff7f0e']  # Use phase color scheme for both figures

    # Extend frequencies and data to reach 1.0
    for i in range(3):
        frc_data[i]['frequencies'], frc_data[i] = extend_to_one(frequencies, frc_data[i])

    # Plot for amplitude
    for i in range(3):
        # Clip the values to a maximum of 1
        mean_amp_clipped = np.clip(frc_data[i]['mean_amp'], 0, 1)
        std_amp_clipped = np.clip(frc_data[i]['std_amp'], 0, 1 - mean_amp_clipped)
        indices = list(range(0, len(frc_data[i]['frequencies']), stdev_step))
        if indices[-1] != len(frc_data[i]['frequencies']) - 1:
            indices.append(len(frc_data[i]['frequencies']) - 1)

        # Plotting amplitude mean and std on the left subplot
        ax_amp.plot(frc_data[i]['frequencies'][indices], mean_amp_clipped[indices], label=labels[2*i], color=phase_colors[i])
        ax_amp.fill_between(frc_data[i]['frequencies'][indices], 
                            mean_amp_clipped[indices] - std_amp_clipped[indices], 
                            mean_amp_clipped[indices] + std_amp_clipped[indices], 
                            color=phase_colors[i], alpha=0.3)
    
    # Plot for phase
    for i in range(3):
        mean_ph_clipped = np.clip(frc_data[i]['mean_ph'], 0, 1)
        std_ph_clipped = np.clip(frc_data[i]['std_ph'], 0, 1 - mean_ph_clipped)
        indices = list(range(0, len(frc_data[i]['frequencies']), stdev_step))
        if indices[-1] != len(frc_data[i]['frequencies']) - 1:
            indices.append(len(frc_data[i]['frequencies']) - 1)

        # Plotting phase mean and std on the right subplot
        ax_ph.plot(frc_data[i]['frequencies'][indices], mean_ph_clipped[indices], label=labels[2*i+1], color=phase_colors[i])
        ax_ph.fill_between(frc_data[i]['frequencies'][indices], 
                           mean_ph_clipped[indices] - std_ph_clipped[indices], 
                           mean_ph_clipped[indices] + std_ph_clipped[indices], 
                           color=phase_colors[i], alpha=0.3)

    # Add threshold lines to both subplots
    ax_amp.axhline(y=threshold, color='black', linestyle='--', label='1/7 Threshold')
    ax_ph.axhline(y=threshold, color='black', linestyle='--', label='1/7 Threshold')

    # Set labels, limits, and legends
    ax_amp.set_ylim(0, 1.005)
    ax_amp.set_xlim(0, 1)
    ax_amp.set_xlabel('Normalized Spatial Frequency', fontsize=15)
    ax_amp.set_ylabel('Mean FRC (Amplitude)', fontsize=15)
    ax_amp.tick_params(axis='both', which='major', labelsize=10)
    ax_amp.tick_params(axis='both', which='minor', labelsize=10)
    # ax_amp.legend(loc=(1.01, 0.), fontsize=12)

    ax_ph.set_ylim(0, 1.005)
    ax_ph.set_xlim(0, 1)
    ax_ph.set_xlabel('Normalized Spatial Frequency', fontsize=15)
    ax_ph.set_ylabel('Mean FRC (Phase)', fontsize=15)
    ax_ph.tick_params(axis='both', which='major', labelsize=10)
    ax_ph.tick_params(axis='both', which='minor', labelsize=10)
    ax_ph.legend(loc=(1.03, 0.), fontsize=12)

    # Add some space between the subplots
    plt.subplots_adjust(wspace=0.8)

    # Adjust the layout to make room for the legends
    plt.tight_layout()
    plt.show()


def phase_line_profile(label, PtychoFormer_pred, ePtychoFormer_pred, ePIE_pred):
    f, ax = plt.subplots(2, 2, figsize=(4,4))
    ax[0, 0].imshow(label[0], vmin=-np.pi, vmax=np.pi)
    ax[0, 1].imshow(PtychoFormer_pred[0], vmin=-np.pi, vmax=np.pi)
    ax[1, 0].imshow(ePtychoFormer_pred[0], vmin=-np.pi, vmax=np.pi)
    ax[1, 1].imshow(ePIE_pred[0], vmin=-np.pi, vmax=np.pi)
    
    mid_y = label[0].shape[0] // 2
    titles = [["Ground Truth", "PtychoFormer"],
          ["ePtychoFormer", "ePIE"]]
    # Add horizontal lines at the middle of each subplot
    for i in range(2):
        for j in range(2):
            if i == 0 and j == 0:  # Ground Truth (label)
                line_color = 'purple'
            elif i == 0 and j == 1:  # PtychoFormer_pred
                line_color = '#38b000'
            elif i == 1 and j == 0:  # ePtychoFormer_pred
                line_color = '#1f77b4'
            elif i == 1 and j == 1:  # ePIE_pred
                line_color = '#ff7f0e'
            # Add horizontal line
            ax[i, j].axhline(y=mid_y, color=line_color, linestyle='-', linewidth=2)
            ax[i, j].set_title(titles[i][j], fontsize=12)
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)
    plt.show()
        
    label_line = label[0, mid_y, :]
    PtychoFormer_line = PtychoFormer_pred[0, mid_y, :]
    ePtychoFormer_line = ePtychoFormer_pred[0, mid_y, :]
    ePIE_line = ePIE_pred[0, mid_y, :]

    plt.figure(figsize=(5, 4))
    plt.plot(label_line, label='Ground Truth', color='purple')
    plt.plot(ePIE_line, label='ePIE', color='#ff7f0e')
    plt.plot(ePtychoFormer_line, label='ePtychoFormer', color='#1f77b4')
    plt.plot(PtychoFormer_line, label='PtychoFormer', color='#38b000')

    plt.title('Line Profile of Different Predictions')
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_probe_C_grids_data(transmission, step):
    Afield, Hole, Hz2, _ = create_probe(0.77)
    n_diff = 3
    diff_size = 128
    patch_size = diff_size + (n_diff - 1) * step

    ph = torch.tensor(transmission[0][0:patch_size, 0:patch_size], dtype=torch.float64)
    amp = torch.tensor(transmission[1][0:patch_size, 0:patch_size], dtype=torch.float64)

    C = amp * torch.exp(1j * ph).to(torch.complex128)

    diff_data = torch.zeros((n_diff * n_diff, patch_size, patch_size), dtype=torch.float64)
    label_data = torch.zeros((2, patch_size, patch_size), dtype=torch.float64)

    label_data[0] = ph
    label_data[1] = amp
    for n in range(n_diff):
        for m in range(n_diff):
            Temp = C[n * step : n * step + diff_size, m * step : m * step + diff_size] * Hole
            Assumption2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Afield * Temp))) * Hz2)))
            diff_data[n * n_diff + m, n * step : n * step + diff_size, m * step : m * step + diff_size] = abs(Assumption2)**2
    return diff_data.float(), label_data