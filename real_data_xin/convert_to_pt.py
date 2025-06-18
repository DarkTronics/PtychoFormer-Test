
import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import re
import glob

def create_ptychography_dataset(image_dir, output_dir, show_preview=False):
    """
    Creates a single ptychography dataset PT file from PNG images.
    Uses images 1-5, 11-15, 21-25, 31-35, 41-45 for a 5×5 diffraction pattern grid.
    
    Args:
        image_dir: Directory containing the numbered PNG images and ground truth
        output_dir: Directory to save the output PT file
        show_preview: Whether to show visualizations of the created dataset
    
    Returns:
        True if dataset was created successfully, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the image indices for the 5×5 grid
    img_indices = list(range(1, 6)) + list(range(11, 16)) + list(range(21, 26)) + list(range(31, 36)) + list(range(41, 46))
    
    folder_name = os.path.basename(image_dir)
    print(f"\nProcessing directory {folder_name}")
    print(f"Looking for images: {img_indices}")
    
    # Load ground truth images
    amplitude_path = os.path.join(image_dir, "absolute.png")
    phase_path = os.path.join(image_dir, "phase.png")
    
    if not os.path.exists(amplitude_path):
        print(f"Error: Amplitude file not found at {amplitude_path}")
        return False
    if not os.path.exists(phase_path):
        print(f"Error: Phase file not found at {phase_path}")
        return False
        
    # Load and convert to tensors
    amplitude_img = Image.open(amplitude_path).convert('L')
    phase_img = Image.open(phase_path).convert('L')
    
    # Convert to numpy arrays and normalize
    amplitude = np.array(amplitude_img).astype(np.float32) / 255.0
    phase = np.array(phase_img).astype(np.float32) / 255.0
    
    # Create label tensor [2, H, W]
    label = torch.tensor(np.stack([amplitude, phase], axis=0), dtype=torch.float32)
    
    # List to store loaded diffraction patterns
    diffraction_patterns = []
    found_images = []
    
    # Load each image
    for img_idx in img_indices:
        # Try different possible extensions
        found = False
        for ext in ['.png', '.PNG']:
            img_path = os.path.join(image_dir, f"{img_idx}{ext}")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img).astype(np.float32) / 255.0
                    diffraction_patterns.append(img_array)
                    found_images.append(img_idx)
                    found = True
                    break
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        if not found:
            print(f"Warning: Could not find image {img_idx} with any extension")
    
    if len(diffraction_patterns) != 25:
        print(f"Warning: Expected 25 patterns but found {len(diffraction_patterns)}")
        if len(diffraction_patterns) < 10:  # Set a minimum threshold
            print("Too few images found. Skipping this directory.")
            return False
        print("The dataset will be created with the available images.")
    
    # Convert to tensor [N, H, W] where N is the number of loaded patterns
    input_tensor = torch.tensor(np.stack(diffraction_patterns, axis=0), dtype=torch.float32)
    
    # Create data dictionary
    data = {
        'input': input_tensor,
        'label': label
    }
    
    # Save to PT file
    output_path = os.path.join(output_dir, f"data_{folder_name}.pt")
    torch.save(data, output_path)
    print(f"Saved dataset to {output_path}")
    print(f"Input shape: {input_tensor.shape}, Label shape: {label.shape}")
    
    # Show preview if requested
    if show_preview:
        # Show a grid of diffraction patterns
        rows, cols = 5, 5
        plt.figure(figsize=(15, 15))
        plt.suptitle(f"Diffraction Patterns for {folder_name}", fontsize=16)
        
        for i in range(min(25, len(diffraction_patterns))):
            plt.subplot(rows, cols, i+1)
            plt.imshow(diffraction_patterns[i], cmap='viridis')
            plt.title(f"Pattern {found_images[i]}")
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"Label Data for {folder_name}", fontsize=16)
        
        plt.subplot(1, 2, 1)
        plt.imshow(amplitude, cmap='viridis')
        plt.title("Amplitude")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(phase, cmap='twilight')
        plt.title("Phase")
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    return True

def process_all_directories(base_dir, output_dir, show_preview=False):
    """
    Process all numbered directories (01-80) in the base directory.
    
    Args:
        base_dir: Base directory containing numbered subdirectories
        output_dir: Directory to save output PT files
        show_preview: Whether to show previews
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all directories that match the pattern (01-80)
    all_dirs = glob.glob(os.path.join(base_dir, "[0-9][0-9]"))
    
    # Sort the directories numerically
    all_dirs.sort(key=lambda x: int(os.path.basename(x)))
    
    print(f"Found {len(all_dirs)} directories to process")
    
    successful = 0
    failed = 0
    
    # Process each directory
    for dir_path in all_dirs:
        dir_name = os.path.basename(dir_path)
        try:
            # Check if the directory name is between 01 and 80
            dir_num = int(dir_name)
            if 1 <= dir_num <= 80:
                print(f"\nProcessing directory {dir_name} ({dir_num}/80)")
                if create_ptychography_dataset(dir_path, output_dir, show_preview):
                    successful += 1
                else:
                    failed += 1
            else:
                print(f"Skipping directory {dir_name} (outside range 01-80)")
        except ValueError:
            print(f"Skipping non-numeric directory {dir_name}")
    
    print(f"\nProcessing complete: {successful} successful, {failed} failed")
    if successful > 0:
        print(f"Successfully created datasets are available in {output_dir}")

if __name__ == "__main__":
    # Update these paths to match your system
    base_dir = r'C:\Users\avang\OneDrive\Desktop\cuda_venv\rawdata'
    output_dir = r'C:\Users\avang\OneDrive\Desktop\cuda_venv\rawdata\ptfiles3'
    
    # Process all available directories from 01 to 80
    process_all_directories(base_dir, output_dir, show_preview=False)
    
    # To process a single directory, you can still use the original function:
    # single_dir = os.path.join(base_dir, "12")
    # create_ptychography_dataset(single_dir, output_dir, show_preview=True)