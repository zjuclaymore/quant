import pandas as pd
import os

file_path = r"E:\1_basement\ml\data\中国A股日行情_AShareEODPrices\20240104.pickle"
if not os.path.exists(file_path):
    # Try an older one if 2024 doesn't exist (based on file list it seems to go up to 2001 in the visible list, but likely goes further)
    # actually the list was truncated. Let's try 20010104 which I saw in the list.
    file_path = r"E:\1_basement\ml\data\中国A股日行情_AShareEODPrices\20010104.pickle"

try:
    data = pd.read_pickle(file_path)
    print(f"File: {file_path}")
    print(f"Type: {type(data)}")
    if isinstance(data, pd.DataFrame):
        print("Columns:", data.columns.tolist())
        print("Head:")
        print(data.head())
    else:
        print(data)
except Exception as e:
    print(f"Error reading pickle: {e}")
