import os
import sys
import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import cupyx.scipy.fft as cufft
import scipy.fft
from scipy.fft import ifft2, fft2, ifftshift, fftshift
scipy.fft.set_global_backend(cufft)

def run_bee_reconstruction(bee_dir, num_iteration=100, progress_bar=True):
    # Change to bee directory
    os.chdir(bee_dir)

    lam = 632.8 * (10**(-9))
    pixelnum = 1600
    pixelsize = 7.4* (10**(-6))
    stepnumber=5
    d = 142 * (10**(-3))
    k = 2 * np.pi / lam
    z1 = 27 * (10**(-3))
    z2 = 10 * (10**(-3))
    ratio=68

    xx = cp.linspace(-pixelsize*pixelnum/2, pixelsize*pixelnum/2, pixelnum)
    yy = cp.linspace(-pixelsize*pixelnum/2, pixelsize*pixelnum/2, pixelnum)
    [XX, YY] = cp.meshgrid(xx, yy)
    fx = cp.linspace(-1/pixelsize/2, 1/pixelsize/2, pixelnum)
    fy = cp.linspace(-1/pixelsize/2, 1/pixelsize/2, pixelnum)
    [Fx, Fy] = cp.meshgrid(fx, fy)

    hole = (cp.sqrt(XX**2+YY**2)<(1.2*10**(-3))).astype(int)

    H1 = cp.exp(1j*k*d*cp.sqrt(1-(lam*Fx)**2-(lam*Fy)**2+0j))
    H2 = cp.exp((-1j*k*d*cp.sqrt(1-(lam*Fx)**2-(lam*Fy)**2+0j)).real)
    H3 = cp.exp(1j*k*z2*cp.sqrt(1-(lam*Fx)**2-(lam*Fy)**2+0j))

    Tx = cp.loadtxt('x.txt', unpack = True).T
    Ty = cp.loadtxt('y.txt', unpack = True).T

    hole = fftshift(ifft2(ifftshift(H3 * (fftshift(fft2(ifftshift(hole)))))))
    r = cp.sqrt(z1**2 + XX**2 + YY**2)
    Afield = hole * cp.exp(1j*k*r) * 0.1
    Assumption1 = cp.ones((6000,6000)).astype(cp.complex128)

    rx=5
    ry=2

    start_time = time.time()
    for i in range(num_iteration):
        for n in range(5):
            for m in range(stepnumber):
                if(progress_bar):
                    sys.stdout.write("\rIteration:"+ str(i+1)+ "  Progress:"+ str((5*n+m+1)/0.25) + "%")
                    sys.stdout.flush()
                filename = str((n+ry)*10+m+1+rx)+".PNG"
                diffraction_image = plt.imread(filename)
                I = 255 * diffraction_image[1031-800:1031+800,1065-800:1065+800]
                I= cp.array(I)
                x1_assumption1 = 1601 - np.round((Tx[(n+ry)*10+m+rx]-Tx[(n+ry)*10+rx])/ratio)
                x2_assumption1 = x1_assumption1 + pixelnum
                y1_assumption1 = 2601 - np.round((Ty[n+ry]-Ty[ry])/ratio)
                y2_assumption1 = y1_assumption1 + pixelnum
                Tem2 = Assumption1[x1_assumption1:x2_assumption1 , y1_assumption1:y2_assumption1]
                assump2_step1 = ifftshift(H1 * fftshift(fft2(ifftshift(Afield * Tem2))))
                Assumption2 = fftshift(ifft2(assump2_step1))
                Renew = cp.sqrt(abs(I)) * cp.exp(1j * cp.angle(Assumption2))
                Reconstruction = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Renew))) * cp.conj(H1))))
                change = Reconstruction - Afield * Tem2
                Tem2 = Tem2 + cp.conj(Afield) / (abs(Afield)**2 + 1) * abs(Afield) / abs(Afield).max() * change
                temp=Tem2
                Afield = Afield + cp.conj(temp) / (abs(temp)**2 + 0.1) * abs(temp) / abs(temp).max() * change
                Assumption1[x1_assumption1:x2_assumption1 , y1_assumption1:y2_assumption1] = Tem2

    end_time = time.time()
    plt.imsave('absolute.png', abs(Assumption1[2150:2450,3150:3450]).get(), cmap='gray')
    plt.imsave('phase.png', cp.angle(Assumption1[2150:2450,3150:3450]).get(), cmap='gray')
    plt.imsave('probe.png', abs(Afield.get()), cmap='gray')
    print(f"\nDone {bee_dir} in {end_time-start_time:.2f} seconds.")

base_path = r'c:\Users\avang\OneDrive\Desktop\cuda_venv\rawdata' # Path where all bee directories are located
for i in range(1, 81):
    bee_dir = os.path.join(base_path, f"{i:02d}")
    if os.path.isdir(bee_dir):
        run_bee_reconstruction(bee_dir)
    else:
        print(f"Skipping {bee_dir}, does not exist.")