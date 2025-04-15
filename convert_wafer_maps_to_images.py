import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Config
PKL_PATH = '/Users/jackp/573/final/wm811k_data/LSWMD.pkl'
OUTPUT_DIR = 'data'
TARGET_SIZE = (128, 128)
NUM_SAMPLES = 10000  # total number of samples desired

def is_defective(failure):
    """
    Returns True if `failure` indicates a defect.
    Considers failure as non-defective if it is None or empty.
    """
    if failure is None:
        return False
    if isinstance(failure, (list, np.ndarray)) and len(failure) == 0:
        return False
    return True

def setup_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, 'defective'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'non_defective'), exist_ok=True)

def save_resized_image(wafer_map, label, filename):
    # Create a temporary image from the wafer map using matplotlib
    fig = plt.figure(figsize=(2, 2), dpi=100)
    plt.imshow(wafer_map, cmap='gray', interpolation='nearest')
    plt.axis('off')
    
    temp_path = 'temp_wafer.png'
    fig.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Use PIL to open the temp image, convert to RGB and resize it
    with Image.open(temp_path) as img:
        img = img.convert('RGB')
        img_resized = img.resize(TARGET_SIZE)
        save_path = os.path.join(OUTPUT_DIR, label, filename)
        img_resized.save(save_path)

def main():
    setup_dirs()
    print("Loading DataFrame...")
    df = pd.read_pickle(PKL_PATH)
    
    # Add an 'is_defective' column based on the helper function.
    df['is_defective'] = df['failureType'].apply(is_defective)
    
    # Separate into defective and non-defective subsets.
    df_defective = df[df['is_defective']]
    df_nondefective = df[~df['is_defective']]
    
    print("Found {} defective wafers and {} non-defective wafers.".format(
        len(df_defective), len(df_nondefective)))
    
    # Determine the number of samples per class.
    num_defective = NUM_SAMPLES // 2
    num_nondefective = NUM_SAMPLES - num_defective
    
    # Sample each subset (if there aren’t enough samples in one class, take as many as possible).
    df_defective_sampled = df_defective.sample(n=min(num_defective, len(df_defective)), random_state=42)
    df_nondefective_sampled = df_nondefective.sample(n=min(num_nondefective, len(df_nondefective)), random_state=42)
    
    # For clarity, maintain separate counters for each class.
    defective_count = 0
    nondefective_count = 0
    
    print("Processing and saving images...")
    # Process defective samples.
    for _, row in tqdm(df_defective_sampled.iterrows(), total=len(df_defective_sampled), desc="Defective"):
        wafer_map = row['waferMap']
        if wafer_map is None or np.all(np.array(wafer_map) == 0):
            continue  # skip invalid or empty wafer maps
        filename = f"defective_{defective_count:05}.png"
        save_resized_image(np.array(wafer_map), 'defective', filename)
        defective_count += 1

    # Process non-defective samples.
    for _, row in tqdm(df_nondefective_sampled.iterrows(), total=len(df_nondefective_sampled), desc="Non-defective"):
        wafer_map = row['waferMap']
        if wafer_map is None or np.all(np.array(wafer_map) == 0):
            continue  # skip invalid or empty wafer maps
        filename = f"nondefective_{nondefective_count:05}.png"
        save_resized_image(np.array(wafer_map), 'non_defective', filename)
        nondefective_count += 1

    if os.path.exists("temp_wafer.png"):
        os.remove("temp_wafer.png")
    
    total = defective_count + nondefective_count
    print(f"✅ Done! Saved {total} wafer images to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()