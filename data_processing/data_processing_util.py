import os
import sys
import cv2
import torch
import numpy as np
from glob import glob
from PIL import Image
from scipy.fft import ifft2, fft2, ifftshift, fftshift

def central_square(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        square_size = min(width, height)

        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
        return img.crop((left, top, right, bottom))

def load_flickr(dir, memmap_path, img_size):
    raw_imgs_paths = glob("{}/*.jpg".format(dir), recursive=True)
    num_images = len(raw_imgs_paths)
    memmap_images = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(num_images, img_size, img_size))
    for idx, img_dir in enumerate(raw_imgs_paths):
        sys.stdout.write("\rData "+ str(idx))
        sys.stdout.flush()
        img = central_square(img_dir)
        img = img.convert('L').resize((img_size, img_size), resample=Image.LANCZOS)
        img = np.array(img).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        memmap_images[idx] = (img - img.min()) / (img.max() - img.min())
    memmap_images.flush()
    print("\nLoaded Flickr30k and saved to memmap.")
    return memmap_images

def preprocess_flickr(dir, destination):
    '''
    Preprocess Flickr30k images

    Inputs:
    - dir: path to Flickr30k dataset
    - destination: path to store the preprocessed images
    '''
    path = os.path.join(destination, 'preprocess')
    if not os.path.exists(path):
        os.makedirs(path)
    raw_img_size = 500
    memmap_path = os.path.join(path, 'raw_images_memmap.npy')
    raw_imgs = load_flickr(dir, memmap_path, raw_img_size)
    print("\nLoaded Flickr30k")
    num_images = raw_imgs.shape[0]
    indices = np.arange(num_images)
    np.random.shuffle(indices)

    # Split into training and test sets
    split_index = int(num_images * 0.9)  # 90% for training, 10% for testing
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # Use the indices to split the images
    train_images = raw_imgs[train_indices]
    test_images = raw_imgs[test_indices]
    print("Creating training images")
    # data augmentation
    aug_train_images_path = os.path.join(path, "train_images_memmap.npy")
    aug_train_images = np.memmap(aug_train_images_path, dtype='float32', mode='w+', shape=(5 * split_index, raw_img_size, raw_img_size))
    trn_data_idx = 0
    for img in train_images:
        sys.stdout.write("\rTrain Data "+ str(trn_data_idx))
        sys.stdout.flush()
        aug_train_images[trn_data_idx] = img
        trn_data_idx += 1
        aug_train_images[trn_data_idx] = np.rot90(img, 2)
        trn_data_idx += 1
        aug_train_images[trn_data_idx] = np.flipud(img)
        trn_data_idx += 1
        aug_train_images[trn_data_idx] = np.fliplr(img)
        trn_data_idx += 1
        aug_train_images[trn_data_idx] = np.rot90(np.fliplr(img), 2)
        trn_data_idx += 1
    print("\nSaving preprocessed training images...\n")
    aug_train_images.flush() 
    del aug_train_images
    print("Creating test images")
    aug_test_images_path = os.path.join(path, "test_images_memmap.npy")
    aug_test_images = np.memmap(aug_test_images_path, dtype='float32', mode='w+', shape=(5 * (num_images - split_index), raw_img_size, raw_img_size))

    tst_data_idx = 0
    for img in test_images:
        sys.stdout.write("\rTest Data "+ str(tst_data_idx))
        sys.stdout.flush()
        aug_test_images[tst_data_idx] = img
        tst_data_idx += 1
        aug_test_images[tst_data_idx] = np.rot90(img, 2)
        tst_data_idx += 1
        aug_test_images[tst_data_idx] = np.flipud(img)
        tst_data_idx += 1
        aug_test_images[tst_data_idx] = np.fliplr(img)
        tst_data_idx += 1
        aug_test_images[tst_data_idx] = np.rot90(np.fliplr(img), 2)
        tst_data_idx += 1
    aug_test_images.flush()  # Make sure data is written to disk
    print("\nSaving preprocessed test images...")
    del aug_test_images

def create_flickr_transmission(dir):
    raw_img_size = 500
    num_flickr_img = 31782
    train_len = 5*int(num_flickr_img * 0.9)
    test_len = 5*int(num_flickr_img * 0.1)
    train_images = np.memmap(os.path.join(dir, "train_images_memmap.npy"), dtype='float32', mode='r', shape=(train_len, raw_img_size, raw_img_size))
    test_images = np.memmap(os.path.join(dir, "test_images_memmap.npy"), dtype='float32', mode='r', shape=(test_len, raw_img_size, raw_img_size))
    
    print("Loaded raw training and testing images")

    train_indices = np.arange(train_len)
    test_indices = np.arange(test_len)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    # Create reshaped memmap data for transmission functions
    train_image_len = train_len // 2
    test_image_len = test_len // 2

    raw_trn_data_path = os.path.join(dir, 'train_transmission_data_memmap.npy')
    raw_test_data_path = os.path.join(dir, 'test_transmission_data_memmap.npy')

    raw_trn_data = np.memmap(raw_trn_data_path, dtype='float32', mode='w+', shape=(train_image_len, 2, raw_img_size, raw_img_size))
    raw_test_data = np.memmap(raw_test_data_path, dtype='float32', mode='w+', shape=(test_image_len, 2, raw_img_size, raw_img_size))

    # Populate reshaped memmap with shuffled data
    print("\nCreating reshaped train data")
    for i in range(train_image_len):
        sys.stdout.write(f"\rReshaping Train Data {i + 1}/{train_image_len}")
        sys.stdout.flush()
        images = train_images[train_indices[i * 2: (i + 1) * 2]]

        # Modify pixel range for amplitude and phase
        images[1] = images[1] * (1 - 0.05) + 0.05  # Modify amplitude
        images[0] = np.pi * (2 * images[0] - 1)  # Modify phase

        # Assign modified images to reshaped data
        raw_trn_data[i] = images

    raw_trn_data.flush()

    print("\nCreating reshaped test data")
    for i in range(test_image_len):
        sys.stdout.write(f"\rReshaping Test Data {i + 1}/{test_image_len}")
        sys.stdout.flush()
        images = test_images[test_indices[i * 2: (i + 1) * 2]]

        # Modify pixel range for amplitude and phase
        images[1] = images[1] * (1 - 0.05) + 0.05  # Modify amplitude
        images[0] = np.pi * (2 * images[0] - 1)  # Modify phase

        # Assign modified images to reshaped data
        raw_test_data[i] = images

    raw_test_data.flush()

    print("Number of transmission function (Train):", train_image_len)
    print("Number of transmission function (Test):", test_image_len)

def create_probe(r1=1):
    lam = 632.8*(10**(-9))
    pixelnum = 128
    pixelsize = 7.4*(10**(-6))
    d1 = 10*(10**(-3))
    d2 = 20*(10**(-3))
    RI = r1
    Freqency = 1/pixelsize
    Fxvector = np.linspace(-Freqency/2,Freqency/2,pixelnum)
    Fyvector = np.linspace(-Freqency/2,Freqency/2,pixelnum)
    [FxMat, FyMat] = np.meshgrid(Fxvector, Fyvector)
    xx = np.linspace(-pixelsize*pixelnum/2,pixelsize*pixelnum/2,pixelnum)
    yy = np.linspace(-pixelsize*pixelnum/2,pixelsize*pixelnum/2,pixelnum)
    [XX, YY] = np.meshgrid(xx,yy)

    Hole = (np.sqrt(XX**2+YY**2)<=3*10**(-4)).astype(float)
    Hz1 = np.exp((1j*2*np.pi*d1*RI/lam)*(1-(lam*FxMat/RI)**2-(lam*FyMat/RI)**2)**0.5)
    Hz2 = np.exp((1j*2*np.pi*d2*RI/lam)*(1-(lam*FxMat/RI)**2-(lam*FyMat/RI)**2)**0.5)
    amplitude1 = np.ones((pixelnum,pixelnum))
    phase1 = np.zeros((pixelnum,pixelnum))
    wave1 = amplitude1*np.exp(1j*phase1)
    Wave1 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Hole)))*wave1)))
    Afield = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Wave1)))*Hz1)))
    return Afield, Hole.astype(int), Hz2


def create_grid_flickr(dir, destination, save = True):
    '''
    Create dataset of varying grid scans

    Inputs:
    - dir: path to 'preprocess' folder created via 'preprocess_flickr' function
    - destination: path to store the dataset
    '''
    if save:
        os.makedirs(destination, exist_ok=True)
        os.makedirs(os.path.join(destination, 'data'), exist_ok=True)
    raw_img_size = 500
    num_flickr_img = 31782
    train_len = 5*int(num_flickr_img * 0.9)
    test_len = 5*int(num_flickr_img * 0.1)
    train_image_len = train_len // 2
    test_image_len = test_len // 2

    raw_trn_data = np.memmap(os.path.join(dir, "train_transmission_data_memmap.npy"), dtype='float32', mode='r', shape=(train_image_len, 2, raw_img_size, raw_img_size))
    raw_test_data = np.memmap(os.path.join(dir, "test_transmission_data_memmap.npy"), dtype='float32', mode='r', shape=(test_image_len, 2, raw_img_size, raw_img_size))
    
    num_row = 3 # for 3x3 grid diffraction pattern
    diffraction_size = 128
    trn_reverse_idx = train_image_len - 1
    test_reverse_idx = test_image_len - 1

    data_per_cls = 6800
    trn_data_per_cls = int(data_per_cls*0.9)
    tst_data_per_cls = data_per_cls - trn_data_per_cls

    trn_index_assignment = np.arange(train_image_len)
    np.random.shuffle(trn_index_assignment)

    test_index_assignment = np.arange(test_image_len)
    np.random.shuffle(test_index_assignment)

    data_idx = 0
    raw_trn_img_idx = 0
    raw_tst_img_idx = 0
    Afield, Hole, Hz2 = create_probe()
    print(f'Total Dataset Size: {data_per_cls*35}')
    print(f'Training Dataset Size: {trn_data_per_cls*35}')
    print(f'Test Dataset Size: {tst_data_per_cls*35}')
    print(f'Data per Class: {data_per_cls}')
    print(f'Training Data per Class: {trn_data_per_cls}')
    print(f'Test Data per Class: {tst_data_per_cls}')
    print("******************************************")
    for cls_num in range(20, 55+1):  # 20 - 55 pixel lateral offsets

        # Compute maximum number of 3x3 diffraction samples we can create from each transmission
        patch_size = diffraction_size + ((num_row - 1)* cls_num)
        n_row = raw_img_size//patch_size
        n_col = raw_img_size//patch_size
        total_patch = n_row * n_col

        # Create label mask
        label_mask = np.zeros((patch_size, patch_size))
        for n in range(num_row):
            for m in range(num_row):
                label_mask[n*cls_num : n*cls_num+diffraction_size, m*cls_num : m*cls_num+diffraction_size] = np.where(
                    label_mask[n*cls_num : n*cls_num+diffraction_size, m*cls_num : m*cls_num+diffraction_size] == 0,
                    Hole,
                    label_mask[n*cls_num : n*cls_num+diffraction_size, m*cls_num : m*cls_num+diffraction_size])
        
        num_cls_data = 0
        num_raw_data_needed = int(np.ceil(data_per_cls/total_patch)) + 1
        num_selected_patch =  int(total_patch - (total_patch * num_raw_data_needed - data_per_cls)//num_raw_data_needed)
        print(f'Processing {num_raw_data_needed} Raw Data for Class {cls_num}')
        print(f'Index: {data_idx} : {data_idx + data_per_cls}')
        trn_to_test_transition = False
        for _ in range(num_raw_data_needed):
            if(num_cls_data < trn_data_per_cls):
                phase, amp = raw_trn_data[trn_index_assignment[raw_trn_img_idx]]
                raw_trn_img_idx += 1
            else:
                phase, amp = raw_test_data[test_index_assignment[raw_tst_img_idx]]
                raw_tst_img_idx += 1
            patch_idx = np.arange(total_patch)
            np.random.shuffle(patch_idx)
            selected_patch_idx = patch_idx[:num_selected_patch]

            # divide images into patches
            phase_patches = [phase[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
            amp_patches = [amp[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
            
            diff_data = np.zeros((num_row*num_row, patch_size, patch_size))
            label_data = np.zeros((2, patch_size, patch_size))

            # Iterate through the patches
            for idx in selected_patch_idx:
                # Mid patch iteration, break if raw image should change to test image or we generated enough data per class
                if((trn_to_test_transition == False) and (num_cls_data == trn_data_per_cls)): 
                    trn_to_test_transition = True
                    break
                if(num_cls_data == data_per_cls):
                    break

                phase_patch = phase_patches[idx]
                amp_patch = amp_patches[idx]

                # If faulty image, replace it
                while(phase_patch.max() == phase_patch.min()):
                    if(num_cls_data <= trn_data_per_cls):
                        phase, _ = raw_trn_data[trn_index_assignment[trn_reverse_idx]]
                        trn_reverse_idx -= 1
                    else:
                        phase, _ = raw_test_data[test_index_assignment[test_reverse_idx]]
                        test_reverse_idx -= 1
                    phase_patches = [phase[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
                    rand_idx = np.random.choice(patch_idx)
                    phase_patch = phase_patches[rand_idx]

                while(amp_patch.max() == amp_patch.min()):
                    if(num_cls_data <= trn_data_per_cls):
                        _, amp = raw_trn_data[trn_index_assignment[trn_reverse_idx]]
                        trn_reverse_idx -= 1
                    else:
                        _, amp = raw_test_data[test_index_assignment[test_reverse_idx]]
                        test_reverse_idx -= 1
                    amp_patches = [amp[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
                    rand_idx = np.random.choice(patch_idx)
                    amp_patch = amp_patches[rand_idx]

                # Apply label masking
                label_data[0] = label_mask * phase_patch
                label_data[1] = label_mask * amp_patch

                # Create complex amp
                ph, amp = np.array(phase_patch), np.array(amp_patch)
                C = amp * np.exp(1j*ph)

                # Generate diffraction patterns
                for n in range(num_row):
                    for m in range(num_row):
                        Temp = C[n*cls_num : diffraction_size + n*cls_num, m*cls_num : diffraction_size + m*cls_num] * Hole
                        Assumption2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Afield * Temp))) * Hz2)))
                        diff_data[n*num_row + m, n*cls_num : diffraction_size + n*cls_num, m*cls_num : diffraction_size + m*cls_num] = abs(Assumption2)**2
                
                # Save diffraction patterns as input, and masked amplitude/phase as label
                data = {
                    'input': torch.tensor(diff_data).float(),
                    'label': torch.tensor(label_data).float(),
                    }
                torch.save(data, destination + f'/data/data_{data_idx}')

                if(trn_to_test_transition == False):
                    sys.stdout.write("\rTrn Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                    sys.stdout.flush()
                elif((trn_to_test_transition == True) and (num_cls_data == trn_data_per_cls)):
                    print("\n")
                    sys.stdout.write("\rTest Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                    sys.stdout.flush()
                else:
                    sys.stdout.write("\rTest Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                    sys.stdout.flush()
                data_idx += 1
                num_cls_data += 1


def create_single_flickr(dir, destination, save = True):
    '''
    Create dataset of single diffraction patterns. This function follows the 
    same scheme as create_grid_flickr(), but creates 9 separate labeled data
    instead of 3x3 grid data.

    Inputs:
    - dir: path to 'preprocess' folder created via 'preprocess_flickr' function
    - destination: path to store the dataset
    '''
    if save:
        os.makedirs(destination, exist_ok=True)
        os.makedirs(os.path.join(destination, 'data'), exist_ok=True)
    raw_img_size = 500
    num_flickr_img = 31782
    train_len = 5*int(num_flickr_img * 0.9)
    test_len = 5*int(num_flickr_img * 0.1)
    train_image_len = train_len // 2
    test_image_len = test_len // 2

    raw_trn_data = np.memmap(os.path.join(dir, "train_transmission_data_memmap.npy"), dtype='float32', mode='r', shape=(train_image_len, 2, raw_img_size, raw_img_size))
    raw_test_data = np.memmap(os.path.join(dir, "test_transmission_data_memmap.npy"), dtype='float32', mode='r', shape=(test_image_len, 2, raw_img_size, raw_img_size))

    num_row = 3
    diffraction_size = 128
    trn_reverse_idx = train_image_len - 1
    test_reverse_idx = test_image_len - 1

    data_per_cls = 6800 * num_row * num_row
    trn_data_per_cls = int(data_per_cls*0.9)
    tst_data_per_cls = data_per_cls - trn_data_per_cls

    trn_index_assignment = np.arange(train_image_len)
    np.random.shuffle(trn_index_assignment)

    test_index_assignment = np.arange(test_image_len)
    np.random.shuffle(test_index_assignment)


    data_idx = 0
    raw_trn_img_idx = 0
    raw_tst_img_idx = 0
    Afield, Hole, Hz2 = create_probe()
    print(f'Total Dataset Size: {data_per_cls*35}')
    print(f'Training Dataset Size: {trn_data_per_cls*35}')
    print(f'Test Dataset Size: {tst_data_per_cls*35}')
    print(f'Data per Class: {data_per_cls}')
    print(f'Training Data per Class: {trn_data_per_cls}')
    print(f'Test Data per Class: {tst_data_per_cls}')
    print("******************************************")
    for cls_num in range(20, 55+1):  # 20 - 55 pixel lateral offsets

        # Compute maximum number of 3x3 diffraction samples we can create from each transmission
        patch_size = diffraction_size + ((num_row - 1)* cls_num)
        n_row = raw_img_size//patch_size
        n_col = raw_img_size//patch_size
        total_patch = n_row * n_col
        
        num_cls_data = 0
        num_raw_data_needed = int(np.ceil(data_per_cls/total_patch))
        num_selected_patch =  int(total_patch - (total_patch * num_raw_data_needed - data_per_cls)//num_raw_data_needed)
        print(f'Processing {num_raw_data_needed} Raw Data for Class {cls_num}')
        print(f'Index: {data_idx} : {data_idx + 9*data_per_cls}')
        trn_to_test_transition = False
        for _ in range(num_raw_data_needed):
            if(num_cls_data < trn_data_per_cls):
                phase, amp = raw_trn_data[trn_index_assignment[raw_trn_img_idx]]
                raw_trn_img_idx += 1
            else:
                phase, amp = raw_test_data[test_index_assignment[raw_tst_img_idx]]
                raw_tst_img_idx += 1
            patch_idx = np.arange(total_patch)
            np.random.shuffle(patch_idx)
            selected_patch_idx = patch_idx[:num_selected_patch]

            # divide images into patches
            phase_patches = [phase[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
            amp_patches = [amp[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
            
            diff_data = np.zeros((num_row*num_row, 1, diffraction_size, diffraction_size))
            label_data = np.zeros((num_row*num_row, 2, diffraction_size, diffraction_size))

            # Iterate through the patches
            for idx in selected_patch_idx:
                # Mid patch iteration, break if raw image should change to test image or we generated enough data per class
                if((trn_to_test_transition == False) and (num_cls_data == trn_data_per_cls)): 
                    trn_to_test_transition = True
                    break
                if(num_cls_data == data_per_cls):
                    break

                phase_patch = phase_patches[idx]
                amp_patch = amp_patches[idx]

                # If faulty image, replace it
                while(phase_patch.max() == phase_patch.min()):
                    if(num_cls_data <= trn_data_per_cls):
                        phase, _ = raw_trn_data[trn_index_assignment[trn_reverse_idx]]
                        trn_reverse_idx -= 1
                    else:
                        phase, _ = raw_test_data[test_index_assignment[test_reverse_idx]]
                        test_reverse_idx -= 1
                    phase_patches = [phase[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
                    rand_idx = np.random.choice(patch_idx)
                    phase_patch = phase_patches[rand_idx]

                while(amp_patch.max() == amp_patch.min()):
                    if(num_cls_data <= trn_data_per_cls):
                        _, amp = raw_trn_data[trn_index_assignment[trn_reverse_idx]]
                        trn_reverse_idx -= 1
                    else:
                        _, amp = raw_test_data[test_index_assignment[test_reverse_idx]]
                        test_reverse_idx -= 1
                    amp_patches = [amp[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
                    rand_idx = np.random.choice(patch_idx)
                    amp_patch = amp_patches[rand_idx]

                # Create complex amp
                ph, amp = np.array(phase_patch), np.array(amp_patch)
                C = amp * np.exp(1j*ph)

                # Generate diffraction patterns and masked labels
                for n in range(num_row):
                    for m in range(num_row):
                        Temp = C[n*cls_num : diffraction_size + n*cls_num, m*cls_num : diffraction_size + m*cls_num] * Hole
                        Assumption2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Afield * Temp))) * Hz2)))
                        diff_data[n*num_row + m, 0] = np.sqrt(abs(Assumption2)**2)
                        label_data[n*num_row + m, 0] = phase_patch[n*cls_num : diffraction_size + n*cls_num, m*cls_num : diffraction_size + m*cls_num] * Hole
                        label_data[n*num_row + m, 1] = amp_patch[n*cls_num : diffraction_size + n*cls_num, m*cls_num : diffraction_size + m*cls_num] * Hole
                
                # Save diffraction patterns as input, and masked amplitude/phase as label
                for i in range(num_row*num_row):
                    data = {
                        'input': torch.tensor(diff_data[i]).float(),
                        'label': torch.tensor(label_data[i]).float(),
                        }
                    torch.save(data, destination + f'/data/data_{data_idx}')
                    if(trn_to_test_transition == False):
                        sys.stdout.write("\rTrn Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                        sys.stdout.flush()
                    elif((trn_to_test_transition == True) and (num_cls_data == trn_data_per_cls)):
                        print("\n")
                        sys.stdout.write("\rTest Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                        sys.stdout.flush()
                    else:
                        sys.stdout.write("\rTest Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                        sys.stdout.flush()
                    num_cls_data += 1
                    data_idx += 1


def create_mask(cls_num):
    '''
    Create masks used for label masking. This function generates masks for 

    Inputs:
    - dir: path to 'preprocess' folder created via 'preprocess_flickr' function
    - destination: path to store the dataset
    '''
    pixelnum = 128
    pixelsize = 7.4 * (10**(-6))
    xx = np.linspace(-pixelsize * pixelnum / 2, pixelsize * pixelnum / 2, pixelnum)
    yy = np.linspace(-pixelsize * pixelnum / 2, pixelsize * pixelnum / 2, pixelnum)
    XX, YY = np.meshgrid(xx, yy)
    Hole = (np.sqrt(XX**2 + YY**2) <= 3 * 10**(-4)).astype(float)
    # Pattern parameters
    diffraction_size = 128
    rand_offset = None
    mask = np.zeros((200 , 200))
    
    if cls_num == 0:
        # 5 diffraction
        coord = np.array([
            [10, 20],
            [10, 20+32],
            [100-64, 100-64],
            [200-128-10, 20],
            [200-128-10, 20+32],
        ])
        for n in range(3):
            for m in range(3):
                if(3*n+m >= coord.shape[0]):
                    break
                mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] = np.where(
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] == 0,
                    Hole,
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size]
                )
    elif cls_num == 1:
        # 6 diffraction
        coord = np.array([
            [5, 18],
            [5, 20+40],
            [50, 4],
            [36, 36],
            [50, 68],
            [72, 36],
        ])
        for n in range(3):
            for m in range(3):
                if(3*n+m >= coord.shape[0]):
                    break
                mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] = np.where(
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] == 0,
                    Hole,
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size]
                )
    elif cls_num == 2:
        # 7 diffraction
        coord = np.array([
            [0, 20],
            [0, 20+36],
            [36, 0],
            [36, 36],
            [36, 72],
            [72, 20],
            [72, 20+36],
        ])
        for n in range(3):
            for m in range(3):
                if(3*n+m >= coord.shape[0]):
                    break
                mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] = np.where(
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] == 0,
                    Hole,
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size]
                )

    elif cls_num == 3:
        # 8 diffraction
        coord = np.array([
            [0, 36],
            [20, 6],
            [36, 36],
            [20, 72-6],
            [48, 0],
            [48, 72],
            [72, 20],
            [72, 20+36],
        ])
        for n in range(3):
            for m in range(3):
                if(3*n+m >= coord.shape[0]):
                    break
                mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] = np.where(
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] == 0,
                    Hole,
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size]
                )
    if cls_num == 4:
        # 9 diffraction diamond layout
        coord = np.array([
            [57-11-11+1, 0+1],
            [85-11-11-11+1, 28-11+1],
            [114-11-11-11-11+1, 57-11-11+1],
            [29-11+1, 28-11+1],
            [57-11-11+1, 57-11-11+1],
            [85-11-11-11+1, 85-11-11-11+1],
            [0+1, 57-11-11+1],
            [29-11+1, 85-11-11-11+1],
            [57-11-11+1, 113-11-11-11-11+1],
        ])

        for n in range(3):
            for m in range(3):
                mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] = np.where(
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] == 0,
                    Hole,
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size]
                )
    elif cls_num == 5:
        #9 diffraction parallelogram layout
        coord = np.array([
            [0, 0],
            [21, 0],
            [42, 0],
            [15+0, 36],
            [15+21, 36],
            [15+42, 36],
            [2*15+0, 72],
            [2*15+21, 72],
            [2*15+42, 72],
        ])
        for n in range(3):
            for m in range(3):
                mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] = np.where(
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size] == 0,
                    Hole,
                    mask[coord[3*n+m,0] : coord[3*n+m,0] + diffraction_size, coord[3*n+m,1] : coord[3*n+m,1] + diffraction_size]
                )
    elif cls_num == 6:
        # 9 diffraction random layout 
        step_size = 26
        rand_offset = np.random.randint(-5,6,(9,2))
        for n in range(3):
            for m in range(3):
                mask[n * step_size+5+rand_offset[3*n+m,0] : n * step_size+5+rand_offset[3*n+m,0] + diffraction_size, m * step_size+5+rand_offset[3*n+m,1] : m * step_size+5+rand_offset[3*n+m,1] + diffraction_size] = np.where(
                    mask[n * step_size+5+rand_offset[3*n+m,0] : n * step_size+5+rand_offset[3*n+m,0] + diffraction_size, m * step_size+5+rand_offset[3*n+m,1] : m * step_size+5+rand_offset[3*n+m,1] + diffraction_size] == 0,
                    Hole,
                    mask[n * step_size+5+rand_offset[3*n+m,0] : n * step_size+5+rand_offset[3*n+m,0] + diffraction_size, m * step_size+5+rand_offset[3*n+m,1] : m * step_size+5+rand_offset[3*n+m,1] + diffraction_size]
                )
    elif cls_num == 7:
        # 9 diffraction grid layout 
        step_size = 36
        for n in range(3):
            for m in range(3):
                mask[n * step_size : n * step_size + diffraction_size, m * step_size : m * step_size + diffraction_size] = np.where(
                    mask[n * step_size : n * step_size + diffraction_size, m * step_size : m * step_size + diffraction_size] == 0,
                    Hole,
                    mask[n * step_size : n * step_size + diffraction_size, m * step_size : m * step_size + diffraction_size]
                )
    return mask, rand_offset

def create_new_scans(dir, destination, save=True):
    '''
    Create dataset of new scan patterns, including five to eight diffraction patterns in circular scans,
    and nine diffraction patterns in diamond, parallelogram, constant grid, and random scans. 
    All input and label dimensions are set to 200x200.  

    Inputs:
    - dir: path to 'preprocess' folder created via 'preprocess_flickr' function
    - destination: path to store the dataset
    '''
    if save:
        os.makedirs(destination, exist_ok=True)
        os.makedirs(os.path.join(destination, 'data'), exist_ok=True)
    raw_img_size = 500
    num_flickr_img = 31782
    train_len = 5*int(num_flickr_img * 0.9)
    test_len = 5*int(num_flickr_img * 0.1)
    train_image_len = train_len // 2
    test_image_len = test_len // 2

    raw_trn_data = np.memmap(os.path.join(dir, "train_transmission_data_memmap.npy"), dtype='float32', mode='r', shape=(train_image_len, 2, raw_img_size, raw_img_size))
    raw_test_data = np.memmap(os.path.join(dir, "test_transmission_data_memmap.npy"), dtype='float32', mode='r', shape=(test_image_len, 2, raw_img_size, raw_img_size))

    # raw_data_idx = 62000
    # raw_data_shape = (62000, 2, 500, 500)
    # raw_data = np.memmap('/home/rnakaha2/documents/research/data_generator/3_6/raw_data_pi.dat', dtype='float32', mode='r+', shape=raw_data_shape)
    
    num_row = 3
    num_cls = 8
    diffraction_size = 128
    trn_reverse_idx = train_image_len - 1
    test_reverse_idx = test_image_len - 1
    # reverse_idx = raw_data_idx - 1

    data_per_cls = 2223
    trn_data_per_cls = int(data_per_cls*0.9)
    tst_data_per_cls = data_per_cls - trn_data_per_cls

    trn_index_assignment = np.arange(train_image_len)
    np.random.shuffle(trn_index_assignment)

    test_index_assignment = np.arange(test_image_len)
    np.random.shuffle(test_index_assignment)

    # index_assignment = np.arange(raw_data_idx)
    # np.random.shuffle(index_assignment)

    # label_size = 128
    # diff_data = np.empty([data_per_cls, num_row, num_row, diffraction_size, diffraction_size])
    # label_data = np.empty([data_per_cls, 2, label_size, label_size])
    data_idx = 0
    # raw_data_idx = 0
    raw_trn_img_idx = 0
    raw_tst_img_idx = 0
    Afield, Hole, Hz2 = create_probe()
    print(f'Total Dataset Size: {data_per_cls*num_cls}')
    print(f'Training Dataset Size: {trn_data_per_cls*num_cls}')
    print(f'Test Dataset Size: {tst_data_per_cls*num_cls}')
    print(f'Data per Class: {data_per_cls}')
    print(f'Training Data per Class: {trn_data_per_cls}')
    print(f'Test Data per Class: {tst_data_per_cls}')
    print("******************************************")
    coord = None
    for cls_num in range(num_cls):
        # Compute maximum number of samples we can create from each transmission
        patch_size = 200
        n_row = raw_img_size//patch_size
        n_col = raw_img_size//patch_size
        total_patch = n_row * n_col
        
        num_cls_data = 0
        num_raw_data_needed = int(np.ceil(data_per_cls/total_patch))
        num_selected_patch =  int(total_patch - (total_patch * num_raw_data_needed - data_per_cls)//num_raw_data_needed)
        print(f'Processing {num_raw_data_needed} Raw Data for Class {cls_num}')
        print(f'Index: {data_idx} : {data_idx + data_per_cls}')
        trn_to_test_transition = False
        for _ in range(num_raw_data_needed):
            # phase, amp = raw_data[index_assignment[raw_data_idx]]
            # raw_data_idx += 1
            if(num_cls_data < trn_data_per_cls):
                phase, amp = raw_trn_data[trn_index_assignment[raw_trn_img_idx]]
                raw_trn_img_idx += 1
            else:
                phase, amp = raw_test_data[test_index_assignment[raw_tst_img_idx]]
                raw_tst_img_idx += 1
            patch_idx = np.arange(total_patch)
            np.random.shuffle(patch_idx)
            selected_patch_idx = patch_idx[:num_selected_patch]

            # divide image in patches
            phase_patches = [phase[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
            amp_patches = [amp[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
            
            diff_data = np.zeros((num_row*num_row, patch_size, patch_size))
            label_data = np.zeros((2, patch_size, patch_size))
            # diff_data = np.zeros((num_selected_patch, num_row*num_row, patch_size, patch_size))
            # label_data = np.zeros((num_selected_patch, 2, patch_size, patch_size))
            # label_data = np.zeros((num_selected_patch, 2, patch_size - 2*offset, patch_size - 2*offset))

            # patch_num = 0
            # Iterate through the patches
            for idx in selected_patch_idx:
                # if(num_cls_data == data_per_cls):
                #     break

                # Mid patch iteration, break if raw image should change to test image or we generated enough data per class
                if((trn_to_test_transition == False) and (num_cls_data == trn_data_per_cls)): 
                    trn_to_test_transition = True
                    break
                if(num_cls_data == data_per_cls):
                    break

                phase_patch = phase_patches[idx]
                amp_patch = amp_patches[idx]

                # If faulty image, replace it
                while(phase_patch.max() == phase_patch.min()):
                    if(num_cls_data <= trn_data_per_cls):
                        phase, _ = raw_trn_data[trn_index_assignment[trn_reverse_idx]]
                        trn_reverse_idx -= 1
                    else:
                        phase, _ = raw_test_data[test_index_assignment[test_reverse_idx]]
                        test_reverse_idx -= 1
                    phase_patches = [phase[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
                    rand_idx = np.random.choice(patch_idx)
                    phase_patch = phase_patches[rand_idx]

                while(amp_patch.max() == amp_patch.min()):
                    if(num_cls_data <= trn_data_per_cls):
                        _, amp = raw_trn_data[trn_index_assignment[trn_reverse_idx]]
                        trn_reverse_idx -= 1
                    else:
                        _, amp = raw_test_data[test_index_assignment[test_reverse_idx]]
                        test_reverse_idx -= 1
                    amp_patches = [amp[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
                    rand_idx = np.random.choice(patch_idx)
                    amp_patch = amp_patches[rand_idx]
                
                # Create label mask
                mask, rand_offset = create_mask(cls_num)

                # apply label masking
                label_data[0] = mask * phase_patch
                label_data[1] = mask * amp_patch

                # Create complex amp
                ph, amp = np.array(phase_patch), np.array(amp_patch)
                C = amp * np.exp(1j*ph)
                
                # Define scan coordinates for each scan pattern
                if cls_num == 0:
                    coord = np.array([
                        [10, 20],
                        [10, 20+32],
                        [100-64, 100-64],
                        [200-128-10, 20],
                        [200-128-10, 20+32],
                    ])

                elif cls_num == 1:
                    coord = np.array([
                        [5, 18],
                        [5, 20+40],
                        [50, 4],
                        [36, 36],
                        [50, 68],
                        [72, 36],
                    ])
                    
                elif cls_num == 2:
                    coord = np.array([
                        [0, 20],
                        [0, 20+36],
                        [36, 0],
                        [36, 36],
                        [36, 72],
                        [72, 20],
                        [72, 20+36],
                    ])
                elif cls_num == 3:
                    coord = np.array([
                        [0, 36],
                        [20, 6],
                        [36, 36],
                        [20, 72-6],
                        [48, 0],
                        [48, 72],
                        [72, 20],
                        [72, 20+36],
                    ])
                elif cls_num == 4:
                    coord = np.array([
                        [57-11-11+1, 0+1],
                        [85-11-11-11+1, 28-11+1],
                        [114-11-11-11-11+1, 57-11-11+1],
                        [29-11+1, 28-11+1],
                        [57-11-11+1, 57-11-11+1],
                        [85-11-11-11+1, 85-11-11-11+1],
                        [0+1, 57-11-11+1],
                        [29-11+1, 85-11-11-11+1],
                        [57-11-11+1, 113-11-11-11-11+1],
                    ])                
                elif cls_num == 5:
                    coord = np.array([
                        [0, 0],
                        [21, 0],
                        [42, 0],
                        [15+0, 36],
                        [15+21, 36],
                        [15+42, 36],
                        [2*15+0, 72],
                        [2*15+21, 72],
                        [2*15+42, 72],
                    ])
                
                # Generate diffraction patterns
                if cls_num < 6:
                    for n in range(num_row):
                        for m in range(num_row):
                            if(3*n+m >= coord.shape[0]):
                                break
                            Temp = C[coord[3*n+m,0] : diffraction_size + coord[3*n+m,0], coord[3*n+m,1] : diffraction_size + coord[3*n+m,1]] * Hole
                            Assumption2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Afield * Temp))) * Hz2)))
                            diff_data[n*num_row + m, coord[3*n+m,0] : diffraction_size + coord[3*n+m,0], coord[3*n+m,1] : diffraction_size + coord[3*n+m,1]] = abs(Assumption2)**2
                elif cls_num == 6:
                    step_size = 26
                    for n in range(num_row):
                        for m in range(num_row):
                            Temp = C[n * step_size+5+rand_offset[3*n+m,0] : n * step_size+5+rand_offset[3*n+m,0] + diffraction_size, m * step_size+5+rand_offset[3*n+m,1] : m * step_size+5+rand_offset[3*n+m,1] + diffraction_size] * Hole
                            Assumption2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Afield * Temp))) * Hz2)))
                            diff_data[n*num_row + m, n * step_size+5+rand_offset[3*n+m,0] : n * step_size+5+rand_offset[3*n+m,0] + diffraction_size, m * step_size+5+rand_offset[3*n+m,1] : m * step_size+5+rand_offset[3*n+m,1] + diffraction_size] = abs(Assumption2)**2
                else:
                    step_size = 36
                    for n in range(num_row):
                        for m in range(num_row):
                            Temp = C[n * step_size : n * step_size + diffraction_size, m * step_size : m * step_size + diffraction_size] * Hole
                            Assumption2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Afield * Temp))) * Hz2)))
                            diff_data[n*num_row + m, n * step_size : n * step_size + diffraction_size, m * step_size : m * step_size + diffraction_size] = abs(Assumption2)**2
                
                # Save diffraction patterns as input, and masked amplitude/phase as label
                data = {
                    'input': torch.tensor(diff_data).float(),
                    'label': torch.tensor(label_data).float(),
                    }
                torch.save(data, destination + f'/data/data_{data_idx}')

                if(trn_to_test_transition == False):
                    sys.stdout.write("\rTrn Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                    sys.stdout.flush()
                elif((trn_to_test_transition == True) and (num_cls_data == trn_data_per_cls)):
                    print("\n")
                    sys.stdout.write("\rTest Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                    sys.stdout.flush()
                else:
                    sys.stdout.write("\rTest Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                    sys.stdout.flush()
                data_idx += 1
                num_cls_data += 1


def create_new_probes(dir, destination, save=True):
    if save:
        os.makedirs(destination, exist_ok=True)
        os.makedirs(os.path.join(destination, 'data'), exist_ok=True)
    raw_img_size = 500
    num_flickr_img = 31782
    train_len = 5*int(num_flickr_img * 0.9)
    test_len = 5*int(num_flickr_img * 0.1)
    train_image_len = train_len // 2
    test_image_len = test_len // 2

    raw_trn_data = np.memmap(os.path.join(dir, "train_transmission_data_memmap.npy"), dtype='float32', mode='r', shape=(train_image_len, 2, raw_img_size, raw_img_size))
    raw_test_data = np.memmap(os.path.join(dir, "test_transmission_data_memmap.npy"), dtype='float32', mode='r', shape=(test_image_len, 2, raw_img_size, raw_img_size))
    
    # raw_data_idx = 62000
    # raw_data_shape = (62000, 2, 500, 500)
    # raw_data = np.memmap('/home/rnakaha2/documents/research/data_generator/3_6/raw_data_pi.dat', dtype='float32', mode='r+', shape=raw_data_shape)
    # reverse_idx = raw_data_idx - 1

    num_row = 3 # for 3x3 grid diffraction pattern
    probe_arg = [1.2, 1, 0.85, 0.8, 0.75, 0.7, 0.65, 0.64, 0.62, 0.6, 0.58] # probe parameters
    num_cls = len(probe_arg)
    diffraction_size = 128
    trn_reverse_idx = train_image_len - 1
    test_reverse_idx = test_image_len - 1
    
    data_per_cls = 2223
    trn_data_per_cls = int(data_per_cls*0.9)
    tst_data_per_cls = data_per_cls - trn_data_per_cls

    trn_index_assignment = np.arange(train_image_len)
    np.random.shuffle(trn_index_assignment)

    test_index_assignment = np.arange(test_image_len)
    np.random.shuffle(test_index_assignment)

    # index_assignment = np.arange(raw_data_idx)
    # np.random.shuffle(index_assignment)
    # label_size = 128
    # diff_data = np.empty([data_per_cls, num_row, num_row, diffraction_size, diffraction_size])
    # label_data = np.empty([data_per_cls, 2, label_size, label_size])
    # raw_data_idx = 0

    data_idx = 0
    raw_trn_img_idx = 0
    raw_tst_img_idx = 0
    print(f'Total Dataset Size: {data_per_cls*num_cls}')
    print(f'Training Dataset Size: {trn_data_per_cls*num_cls}')
    print(f'Test Dataset Size: {tst_data_per_cls*num_cls}')
    print(f'Data per Class: {data_per_cls}')
    print(f'Training Data per Class: {trn_data_per_cls}')
    print(f'Test Data per Class: {tst_data_per_cls}')
    print("******************************************")
    step_size = 20 # 20 pixel lateral offsets
    patch_size = diffraction_size + ((num_row - 1)* step_size)

    # Afield, Hole, Hz2 = create_probe(probe_arg[0])

    # Create label mask
    _, Hole, _ = create_probe(probe_arg[0])
    label_mask = np.zeros((patch_size, patch_size))
    for n in range(num_row):
        for m in range(num_row):
            label_mask[n*step_size : n*step_size+diffraction_size, m*step_size : m*step_size+diffraction_size] = np.where(
                label_mask[n*step_size : n*step_size+diffraction_size, m*step_size : m*step_size+diffraction_size] == 0,
                Hole,
                label_mask[n*step_size : n*step_size+diffraction_size, m*step_size : m*step_size+diffraction_size])
        
    for cls_num in range(num_cls): # 8 different scan patterns
        # Compute maximum number of samples we can create from each transmission
        n_row = raw_img_size//patch_size
        n_col = raw_img_size//patch_size
        total_patch = n_row * n_col
        
        num_cls_data = 0
        num_raw_data_needed = int(np.ceil(data_per_cls/total_patch))
        num_selected_patch =  int(total_patch - (total_patch * num_raw_data_needed - data_per_cls)//num_raw_data_needed)
        print(f'Processing {num_raw_data_needed} Raw Data for Class {cls_num}')
        print(f'Index: {data_idx} : {data_idx + data_per_cls}')
        trn_to_test_transition = False
        for _ in range(num_raw_data_needed):
            # phase, amp = raw_data[index_assignment[raw_data_idx]]
            # raw_data_idx += 1
            if(num_cls_data < trn_data_per_cls):
                phase, amp = raw_trn_data[trn_index_assignment[raw_trn_img_idx]]
                raw_trn_img_idx += 1
            else:
                phase, amp = raw_test_data[test_index_assignment[raw_tst_img_idx]]
                raw_tst_img_idx += 1
            patch_idx = np.arange(total_patch)
            np.random.shuffle(patch_idx)
            selected_patch_idx = patch_idx[:num_selected_patch]

            # divide image into patches
            phase_patches = [phase[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
            amp_patches = [amp[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
            
            diff_data = np.zeros((num_row*num_row, patch_size, patch_size))
            label_data = np.zeros((2, patch_size, patch_size))

            # Iterate through the patches
            for idx in selected_patch_idx:
                # if(num_cls_data == data_per_cls):
                #     break

                # Mid patch iteration, break if raw image should change to test image or we generated enough data per class
                if((trn_to_test_transition == False) and (num_cls_data == trn_data_per_cls)): 
                    trn_to_test_transition = True
                    break
                if(num_cls_data == data_per_cls):
                    break

                phase_patch = phase_patches[idx]
                amp_patch = amp_patches[idx]

                 # If faulty image, replace it
                while(phase_patch.max() == phase_patch.min()):
                    if(num_cls_data <= trn_data_per_cls):
                        phase, _ = raw_trn_data[trn_index_assignment[trn_reverse_idx]]
                        trn_reverse_idx -= 1
                    else:
                        phase, _ = raw_test_data[test_index_assignment[test_reverse_idx]]
                        test_reverse_idx -= 1
                    phase_patches = [phase[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
                    rand_idx = np.random.choice(patch_idx)
                    phase_patch = phase_patches[rand_idx]

                while(amp_patch.max() == amp_patch.min()):
                    if(num_cls_data <= trn_data_per_cls):
                        _, amp = raw_trn_data[trn_index_assignment[trn_reverse_idx]]
                        trn_reverse_idx -= 1
                    else:
                        _, amp = raw_test_data[test_index_assignment[test_reverse_idx]]
                        test_reverse_idx -= 1
                    amp_patches = [amp[x:x+patch_size,y:y+patch_size] for x in range(0,patch_size*n_row,patch_size) for y in range(0,patch_size*n_col,patch_size)]
                    rand_idx = np.random.choice(patch_idx)
                    amp_patch = amp_patches[rand_idx]

                # Create probe function
                Afield, Hole, Hz2 = create_probe(probe_arg[cls_num])

                # Apply label masking
                label_data[0] = label_mask * phase_patch
                label_data[1] = label_mask * amp_patch

                # Create complex amp
                ph, amp = np.array(phase_patch), np.array(amp_patch)
                C = amp * np.exp(1j*ph)

                # Generate diffraction patterns
                for n in range(num_row):
                    for m in range(num_row):
                        Temp = C[n*step_size : diffraction_size + n*step_size, m*step_size : diffraction_size + m*step_size] * Hole
                        Assumption2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Afield * Temp))) * Hz2)))
                        diff_data[n*num_row + m, n*step_size : diffraction_size + n*step_size, m*step_size : diffraction_size + m*step_size] = abs(Assumption2)**2

                # Save diffraction patterns as input, and masked amplitude/phase as label
                data = {
                    'input': torch.tensor(diff_data).float(),
                    'label': torch.tensor(label_data).float(),
                    }
                torch.save(data, destination + f'/data/data_{data_idx}')

                if(trn_to_test_transition == False):
                    sys.stdout.write("\rTrn Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                    sys.stdout.flush()
                elif((trn_to_test_transition == True) and (num_cls_data == trn_data_per_cls)):
                    print("\n")
                    sys.stdout.write("\rTest Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                    sys.stdout.flush()
                else:
                    sys.stdout.write("\rTest Progress:"+ str(num_cls_data) + "\t" + str(data_idx))
                    sys.stdout.flush()
                data_idx += 1
                num_cls_data += 1


def load_flowers(dir):
    raw_imgs = glob("{}/*.jpg".format(dir), recursive=True)
    img_size = 168
    imgs = np.empty((len(raw_imgs), img_size, img_size))
    print((len(raw_imgs)))
    for idx, img_dir in enumerate(raw_imgs):
        img = central_square(img_dir)
        img = img.crop((0, 0, 250, 250))
        img = img.convert('L').resize((img_size, img_size), resample=Image.LANCZOS)
        img = np.array(img).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        imgs[idx] = (img - img.min()) / (img.max() - img.min())
        sys.stdout.write("\rData "+ str(idx) + "\t"+str(img.size))
        sys.stdout.flush()
    return imgs

def load_caltech(dir):
    raw_imgs = glob("{}/*/*.jpg".format(dir), recursive=True)
    img_size = 168
    imgs = np.empty((len(raw_imgs), img_size, img_size))
    print((len(raw_imgs)))
    for idx, img_dir in enumerate(raw_imgs):
        img = central_square(img_dir)
        img = img.convert('L').resize((img_size, img_size), resample=Image.LANCZOS)
        img = np.array(img).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        imgs[idx] = (img - img.min()) / (img.max() - img.min())
        sys.stdout.write("\rData "+ str(idx))
        sys.stdout.flush()
    return imgs

def create_flower_or_caltech_dataset(dir, destination, flower, save=True):
    if save:
        os.makedirs(destination, exist_ok=True)
        os.makedirs(os.path.join(destination, 'data'), exist_ok=True)
    if flower:
        # load images from flower102
        imgs = load_flowers(dir)
    else:
        # load images from caltech101
        imgs = load_caltech(dir)

    # randomly assign each image as amplitude or phase
    rand_assignment1 = np.arange(imgs.shape[0])
    rand_assignment2 = np.arange(imgs.shape[0])
    np.random.shuffle(rand_assignment1)
    np.random.shuffle(rand_assignment2)
    rand_img1 = imgs[rand_assignment1]
    rand_img2 = imgs[rand_assignment2]
    raw_data = np.stack((rand_img1, rand_img2), axis=1)

    # adjust amplitude and phase pixel range accordingly
    for i in range(imgs.shape[0]):
        raw_data[i,1] = raw_data[i,1] * (1 - 0.05) + 0.05
        raw_data[i,0] = np.pi * (2 * raw_data[i,0] - 1)
        sys.stdout.write("\rProgress:"+ str(i))
        sys.stdout.flush()

    diffraction_size = 128
    num_row = 3
    step_size = 20
    data_idx = 0
    patch_size = diffraction_size + ((num_row - 1)* step_size)
    Afield, Hole, Hz2 = create_probe()

    # Create label mask
    mask = np.zeros((patch_size, patch_size))
    for n in range(num_row):
        for m in range(num_row):
            mask[n*step_size : n*step_size+diffraction_size, m*step_size : m*step_size+diffraction_size] = np.where(
                mask[n*step_size : n*step_size+diffraction_size, m*step_size : m*step_size+diffraction_size] == 0,
                Hole,
                mask[n*step_size : n*step_size+diffraction_size, m*step_size : m*step_size+diffraction_size])

    diff_data = np.zeros((num_row*num_row, patch_size, patch_size))
    label_data = np.zeros((2, patch_size, patch_size))

    for idx, img in (raw_data):
        if(img[0].min() == img[0].max()):
            print("invalid phase:", idx)
        if(img[1].min() == img[1].max()):
            print("invalid amplitude:", idx)
        ph, amp = img[0], img[1]

        # Apply label mask
        label_data[0] = mask * ph
        label_data[1] = mask * amp

        # Create complex amp
        C = amp * np.exp(1j*ph)

        # Generate diffraction patterns
        for n in range(num_row):
            for m in range(num_row):
                Temp = C[n*step_size : diffraction_size + n*step_size, m*step_size : diffraction_size + m*step_size] * Hole
                Assumption2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Afield * Temp))) * Hz2)))
                diff_data[n*num_row + m, n*step_size : diffraction_size + n*step_size, m*step_size : diffraction_size + m*step_size] = abs(Assumption2)**2
        
        # Save diffraction patterns as input, and masked amplitude/phase as label
        data = {
            'input': torch.tensor(diff_data).float(),
            'label': torch.tensor(label_data).float()
            }
        torch.save(data, destination + f'/data/data_{data_idx}')
        sys.stdout.write("\rProgress:"+ str(data_idx))
        sys.stdout.flush()
        data_idx += 1
