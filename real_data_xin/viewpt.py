import torch

# Path to the .pt file
file_path = '/home/xguo50/PtychoFormer/5by5realdata/ptfiles3/data_01.pt'

# Load the .pt file
try:
    data = torch.load(file_path, map_location='cpu')  # Use 'cpu' to avoid CUDA issues
    print("File loaded successfully!\n")

    # Check type of the loaded object
    print(f"Type of loaded data: {type(data)}\n")

    # If it's a dictionary, print its keys
    if isinstance(data, dict):
        print("Top-level keys in the file:")
        for key in data.keys():
            print(f" - {key}")
        
        # Optionally show what's inside one key
        first_key = next(iter(data))
        print(f"\nExample content from key '{first_key}':\n{data[first_key]}")
    
    # If it's a list or tensor
    elif isinstance(data, list):
        print(f"First item in list:\n{data[0]}")
    elif isinstance(data, torch.Tensor):
        print(f"Tensor shape: {data.shape}\n")
        print(data)
    else:
        print("Data content:")
        print(data)

except Exception as e:
    print(f"Error loading the file: {e}")


    import torch

def summarize_tensor(tensor, name):
    print(f"Summary of '{name}':")
    print(f"  - Shape: {tensor.shape}")
    print(f"  - Dtype: {tensor.dtype}")
    print(f"  - Min: {tensor.min().item():.4f}")
    print(f"  - Max: {tensor.max().item():.4f}")
    print(f"  - Mean: {tensor.mean().item():.4f}")
    print()

def view_pt_file(file_path):
    try:
        data = torch.load(file_path, map_location='cpu')
        print("✅ File loaded successfully!\n")

        print(f"📦 Type of loaded data: {type(data)}\n")

        if isinstance(data, dict):
            print("🔑 Top-level keys:")
            for key in data:
                print(f" - {key}")
            print()

            for key, value in data.items():
                print(f"🔍 Inspecting key: '{key}'")
                if isinstance(value, torch.Tensor):
                    summarize_tensor(value, key)
                elif isinstance(value, (int, float, str)):
                    print(f"  - Value: {value}\n")
                elif isinstance(value, list):
                    print(f"  - List of length {len(value)}")
                    print(f"    First item: {value[0]}\n")
                else:
                    print(f"  - Type: {type(value)}\n")
        else:
            print("⚠️ Loaded data is not a dictionary.")
            print(data)

    except Exception as e:
        print(f"❌ Error loading file: {e}")

# === Run the viewer ===
view_pt_file('/home/xguo50/PtychoFormer/5by5realdata/ptfiles3/data_01.pt')

