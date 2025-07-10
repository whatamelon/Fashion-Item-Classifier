import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# 1. Load test data
test_df = pd.read_csv("sets/train_data.csv")

# 2. Define LABEL_MAP
LABEL_MAP = {
    0: "10-10-11", 1: "10-10-12", 2: "10-10-13", 3: "10-10-14", 4: "10-10-15",
    5: "10-20-21", 6: "10-20-22", 7: "10-20-23", 8: "10-30-31", 9: "10-30-32",
    10: "10-50-00", 11: "20-10-11", 12: "20-10-12", 13: "20-10-13", 14: "20-10-14",
    15: "20-10-15", 16: "20-10-16", 17: "20-20-21", 18: "20-20-23", 19: "20-20-24",
    20: "20-30-31", 21: "20-30-32", 22: "20-30-33", 23: "20-40-41", 24: "20-40-42",
    25: "20-51-51", 26: "20-51-52", 27: "20-51-53", 28: "20-51-54", 29: "20-51-55",
    30: "20-52-00", 31: "20-53-00"
}
LABEL_MAP_REVERSE = {v: k for k, v in LABEL_MAP.items()}

# 3. Prepare output
processed_data = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    cat_code = row['cat_code']
    img_url = row['img_main']
    label = LABEL_MAP_REVERSE.get(cat_code)

    if not label or not isinstance(img_url, str) or not img_url.startswith("http"):
        continue

    try:
        # Download and process image
        response = requests.get(img_url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("L")
        img = img.resize((28, 28))
        pixels = np.array(img).flatten()
        if len(pixels) != 784:
            continue

        processed_data.append([label] + pixels.tolist())

    except Exception as e:
        print(f"[SKIPPED] {img_url} — {e}")
        continue

# 4. Save to CSV
if processed_data:
    columns = ['label'] + [f'pixel{i}' for i in range(1, 785)]
    df_out = pd.DataFrame(processed_data, columns=columns)
    df_out.to_csv("fashion-mnist_test_sample.csv", index=False)
    print("✅ Sample CSV saved to fashion-mnist_test_sample.csv")
else:
    print("⚠️ No data processed.")