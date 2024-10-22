import numpy as np
import time
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from torch.fft import ifft2, fft2, ifftshift, fftshift

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

# def ePIE(diffraction, step, iteration, n_row, Ref, probe_upd = True, record = False, verbose = False, plot_freq = 20, alpha=1, beta=1):
#     '''
#     This function runs the extended-ptychographic iterative engine (ePIE). This implementation only allows for grid scans.

#     Inputs:
#     - diffraction: input diffraction patterns in n x n grid
#     - step: lateral offsets between each scan position
#     - iteration: number of iteration for ePIE
#     - n_row: square root of the number of diffraction patterns (n)
#     - Ref: initial tranmission estimate
#     - probe_upd: if true, update probe function, else don't update probe function  
#     - record: if True, record the sum squared error per iteration
#     - verbose: print the current iteration count and plot the current estimated transmission and probe function
#     - plot_freq: the frequency of plot generation
#     - alpha: update speed of estimated transmission function
#     - beta: update speed of estimated probe function
#     '''
#     a = 0.05
#     b = 0.05
#     pixelnum = 128
#     Wave, Hole, Hz2, _ = create_probe()
#     hist_loss = []
#     new = np.zeros((pixelnum, pixelnum), dtype=np.complex128)
#     start_time = time.time()
#     for iter in range(iteration):
#         if(verbose):
#             sys.stdout.write("\rIteration:"+ str(iter+1))
#             sys.stdout.flush()
#         for m in range(n_row):
#             for n in range(n_row):
#                 amplitude = np.sqrt(diffraction[m * n_row + n]).astype(np.float64)

#                 new = (Ref[m * step : m * step + pixelnum, n * step : n * step + pixelnum] * Hole).astype(np.complex128)
#                 WaveRef = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(new * Wave))) * Hz2)))
#                 phase = np.angle(WaveRef).astype(np.float64)
#                 WaveRef1 = amplitude * np.exp(1j * phase).astype(np.complex128)
             
#                 WaveRefr1 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(WaveRef1))) * np.conj(Hz2))))
#                 change = (WaveRefr1 - Wave * new).astype(np.complex128)
                
#                 new = new + alpha*(change) * ((abs(Wave) * np.conj(Wave)) / (np.amax(abs(Wave)) * (abs(Wave)**2 + a))).astype(np.complex128)
#                 mask = Hole.astype(int) == 1
#                 Ref[m * step : m * step + pixelnum, n * step : n * step + pixelnum][mask] = new[mask]
#                 if probe_upd:
#                     Wave = Wave + beta*((abs(new) * np.conj(new)) / (np.amax(abs(new) * pixelnum) * (abs(new)**2 + b)))*(WaveRefr1 - Wave * new)
#         if(record and ((iter+1) % 2 == 0)):
#             # compute SSE loss
#             SSELoss = []
#             for m in range(n_row):
#                 for n in range(n_row):
#                     truth = np.array(diffraction[m * n_row + n])
#                     new = Ref[m * step : m * step + pixelnum, n * step : n * step + pixelnum] * Hole
#                     guess = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(new * Wave))) * Hz2)))
#                     guess = abs(guess)**2
#                     SSELoss.append(nn.functional.l1_loss(torch.tensor(guess), torch.tensor(truth)).item())
#             hist_loss.append(np.mean(SSELoss))
#             sys.stdout.write("\rIter "+ str(iter+1) + " SSE Loss:"+ str(np.mean(SSELoss)))
#             sys.stdout.flush()
#         if(verbose and ((iter+1) % plot_freq == 0)):
#             f, ax = plt.subplots(1,4, figsize=(15,12))
#             amp = abs(Ref)
#             pha = np.angle(Ref)
#             ax[0].imshow(pha[0:(n_row-1)*step + pixelnum, 0:(n_row-1)*step + pixelnum], cmap='gray')
#             ax[1].imshow(amp[0:(n_row-1)*step + pixelnum, 0:(n_row-1)*step + pixelnum], cmap='gray')
#             ax[2].imshow(abs(Wave), cmap='gray')
#             ax[3].imshow(np.angle(Wave), cmap='gray')
#             plt.show(block=False)
#     total_time = (time.time() - start_time)
#     Amplitude = abs(Ref)
#     Phase = unwrap_phase(np.angle(Ref))
#     return Phase, Amplitude, hist_loss, total_time



def flat_initialization(n_row, step):
    AmplitudeRef = torch.ones((128 + (n_row-1) * step,128 + (n_row-1) * step))
    PhaseRef = torch.zeros((128 + (n_row-1) * step,128 + (n_row-1) * step))
    Ref = AmplitudeRef*torch.exp(1j*PhaseRef).to(torch.complex128)
    return Ref

# def ePIE_torch(diffraction, step, iteration, n_row, tgt_amp, tgt_ph, Ref, record = False, verbose = True, plot_freq = 20, a = 0.05, b = 0.05):
#     device = diffraction.device
#     pixelnum = 128
#     Wave, Hole, Hz2, HzMinus = create_probe()
#     Wave, Hole, Hz2, HzMinus = Wave.to(device), Hole.to(device), Hz2.to(device), HzMinus.to(device)
#     new = torch.zeros((pixelnum,pixelnum)).to(torch.complex128).to(device)
#     hist_loss = []
#     hist_time = []
#     start_time = time.time()
#     with torch.no_grad():
#         for iter in range(iteration):
#             sys.stdout.write("\rIteration:"+ str(iter+1))
#             sys.stdout.flush()
#             start_time1 = time.time()
#             for m in range(n_row):
#                 for n in range(n_row):
#                     amplitude = torch.sqrt(diffraction[m * n_row + n])

#                     new = Ref[m * step : m * step + pixelnum, n * step : n * step + pixelnum] * Hole.to(torch.complex128)
#                     WaveRef = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(new * Wave))) * Hz2)))
            
#                     WaveRef1 = amplitude*torch.exp(1j*torch.angle(WaveRef)).to(torch.complex128)
#                     WaveRefr1 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(WaveRef1)))*HzMinus)))
                    
#                     new = new + ((abs(Wave) * torch.conj(Wave)) / (torch.max(abs(Wave)) * (abs(Wave)**2 + a))) * (WaveRefr1 - Wave * new)
#                     Ref[m * step : m * step + pixelnum, n * step : n * step + pixelnum][Hole == 1] = new[Hole == 1]
#                     # Wave = Wave + ((abs(new) * torch.conj(new)) / (torch.max(abs(new) * pixelnum) * (abs(new)**2 + b)))*(WaveRefr1 - Wave * new)

#                     # Clear unused variables
#                     del amplitude, WaveRef, WaveRef1, WaveRefr1, new
#                     torch.cuda.empty_cache()
#             end_time1 = time.time()
#             total_time1 = (end_time1 - start_time1)
#             hist_time.append(total_time1)
#             if(record):
#                 # compute SSE loss
#                 SSELoss = 0
#                 for m in range(n_row):
#                     for n in range(n_row):
#                         truth = torch.sqrt(diffraction[m * n_row + n])
#                         new = Ref[m * step : m * step + pixelnum, n * step : n * step + pixelnum] * Hole
#                         guess = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(new * Wave))) * Hz2)))
#                         guess = abs(guess)
#                         SSELoss += nn.functional.mse_loss(guess, truth).item()
#                 hist_loss.append(SSELoss)
#     end_time = time.time()
#     total_time = (end_time - start_time)
#     Amplitude = abs(Ref)
#     Phase = torch.angle(Ref)
#     return Phase, Amplitude, hist_loss, total_time, hist_time


def ePIE(diffraction, step, n_row, iteration = 1000, probe_upd = True, record = False, verbose = False, plot_freq = 20, alpha=1, beta=1):
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
    Ref = flat_initialization(n_row, step).to(device)
    a = 0.05
    b = 0.05
    pixelnum = 128
    Wave, Hole, Hz2, HzMinus = create_probe()
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