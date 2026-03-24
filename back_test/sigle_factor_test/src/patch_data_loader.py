import os

file_path = r"E:\1_basement\quant_research\back_test\sigle_factor_test\src\data_loader.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''    base_path = (
        rf"E:\\1_basement\\quant_research\\factor\\{factor_group}\\{factor_name}\\output"
    )'''

replacement = '''    # 优先支持直接路径 (Direct Path support for factor_package structure)
    direct_path = rf"E:\\1_basement\\quant_research\\factor\\{factor_group}\\{factor_name}"
    if os.path.isdir(direct_path):
        import glob
        parquet_files = glob.glob(os.path.join(direct_path, "*.parquet"))
        if parquet_files:
            return direct_path, "parquet"

    base_path = (
        rf"E:\\1_basement\\quant_research\\factor\\{factor_group}\\{factor_name}\\output"
    )'''

if target in content:
    content = content.replace(target, replacement)
    print("Replacement successful")
else:
    # Try with \r\n
    target_crlf = target.replace('\n', '\r\n')
    if target_crlf in content:
        content = content.replace(target_crlf, replacement.replace('\n', '\r\n'))
        print("Replacement successful (CRLF)")
    else:
        print("Target not found exactly.")
        # Let's try finding the function and inserting
        idx = content.find('def get_valid_factor_path(')
        if idx != -1:
            print("Found function, but replacement failed. Manual fallback required.")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
