import cv2
import h5py
import os
from PIL import Image
import numpy as np

# hr_dataset = h5py.File('/home/lyu/lwl_wsp/mmsegmentation/data/LSDSSIMR/2020/20200301/20200301041500.h5')['Others']
# # label = h5py.File('data-train.h5')['train_set_1_tip']
# i = 0
# for member in hr_dataset:
#     print(member)
#     x = hr_dataset[member]
#     y = x[:]

#     cv2.imwrite(f'/home/lyu/lwl_wsp/mmsegmentation/data/LSDSSIMR/label{i}.png', y)
#     i += 1
# # lenght=len(hr_dataset)
# # for i in range(len(hr_dataset)):
# #     y = hr_dataset[i]
# #     # x = label[i]
# #     cv2.imwrite('image_test/%s.png'%i, y)
# #     # cv2.imwrite(str(i)+"0.png", x)
input_folder = '/home/lyu/lwl_wsp/mmsegmentation/data/LSDSSIMR/test'
output_folder = '/home/lyu/lwl_wsp/mmsegmentation/data/LSDSSIMR/test/label'

# Get a list of all .h5 files in the input folder
h5_files = [file for file in os.listdir(input_folder) if file.endswith('.h5')]

for file_name in h5_files:
    h5_file_path = os.path.join(input_folder, file_name)
    hr_dataset = h5py.File(h5_file_path, 'r')['Others']
    # keys = list(hr_dataset.keys())
    # # print(keys)
    # keys_to_take = keys[:3]
    # images = []

    # for key in keys_to_take:
    #     x = hr_dataset[key]
    #     y = x[:]

    #     # Convert the data to an image and append it to the list

    #     images.append(y)
    # y = np.stack(images, axis=-1)
    # print(y.shape)
    
    x = hr_dataset['DST']
    y = x[:]
    

    output_file_path = os.path.join(output_folder, f'{file_name}.png')
    cv2.imwrite(output_file_path, y)
    # np.save(output_file_path, y)
print('over!')