import os
def rename_images(folder_path, file_extension=".png"):

    if not os.path.exists(folder_path):
        print(f"路径 {folder_path} 不存在！")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    files.sort()

    for index, file_name in enumerate(files, start=1):
        old_file_path = os.path.join(folder_path, file_name)
        new_file_name = f"{index}{file_extension}"
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)
        print(f"重命名: {old_file_path} -> {new_file_path}")
    print("重命名完成！")

if __name__ == '__main__':
    folder_path = "./origin_clic2024_numbered"  # 替换为你的文件夹路径
    rename_images(folder_path)