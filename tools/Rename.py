import os

# 定义文件夹路径
folder_path = "/home/lyu/lwl_wsp/mmsegmentation/data/LSDSSIMR/train/label"

# 定义计数器
count = 1

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 如果文件名以".jpg"结尾
    if filename.endswith(".png"):
        # 构造新文件名
        new_filename = f"P00{count}.png"

        # 构造旧文件路径和新文件路径
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)

        # 更新计数器
        count += 1
print('over!')
