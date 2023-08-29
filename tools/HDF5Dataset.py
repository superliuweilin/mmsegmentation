# 开发者：蔡志鹏
# 开发时间：2022/10/24  15:16
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop


def random_rot_flip(image):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    # label = np.rot90(label, k)
    axis = np.random.randint(1, 3)
    image = np.flip(image, axis=axis).copy()
    # label = np.flip(label, axis=axis).copy()
    return image
    # , label


def random_rotate(image):
    # angle = np.random.randint(-20, 20)
    # image = ndimage.rotate(image, angle, order=0, reshape=False)
    image = transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.BILINEAR, )
    # label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image


def left_center_crop(image):
    # np.random.seed(0)
    h = np.random.randint(60, 100)
    w = np.random.randint(130, 300)  # 800
    return crop(image, h, w, 224, 224)


class RandomTransform(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.img_size = (640, 1280)

    def __call__(self, multi_data):
        # transforms.RandomResizedCrop(512, scale=(0.3, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize(mean, std)
        if random.random() > 0.5:
            multi_data = random_rot_flip(multi_data)
            # multi_data = random_rotate(multi_data)



class HDF5Dataset(Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, args, recursive, load_data, data_cache_size=3, transform=None, img_size=(640, 1280)):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.img_size = img_size
        self.args = args

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            # files = sorted(p.glob('**/*.h5'))
            files = sorted(p.glob('**/*.HDF'))
        else:
            # files = sorted(p.glob('*.h5'))
            files = sorted(p.glob('*.HDF'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

        # if 224
        # if img_size[0] == 224:
        #     for r in range(2):
        #         self.data_info.extend(self.data_info)

    def __getitem__(self, index):

        # get data
        # x = self.get_data("data", index)
        # if self.transform:
        #     x = self.transform(x)
        # else:
        #     x = torch.from_numpy(x)
        #
        # # get label
        # y = self.get_data("label", index)
        # y = torch.from_numpy(y)
        # return (x, y)

        # get data
        # multi_data = {'R0.47': self.get_data('R0.47', index), 'R0.65': self.get_data('R0.65', index)}
        get_which = list(['R0.47', 'R0.65', 'R0.83', 'R1.37', 'R1.61', 'R2.22', 'BT3.72', 'BT6.25',
                          'BT7.10', 'BT8.50', 'BT10.8', 'BT12.0', 'BT13.5'])
        multi_data = np.array([self.get_data(get_which[i], index).astype(np.float32) for i in self.args.use_channels])
        label_np = np.array(self.get_data('DST_binary', index))
        label_np = np.expand_dims(label_np, axis=0)
        multi_data = np.append(multi_data, label_np, axis=0)
        # multi_data = np.array([
        #     self.get_data('R0.47', index).astype(np.float32),
        #     self.get_data('R0.65', index).astype(np.float32),
        #     self.get_data('R0.83', index).astype(np.float32),
        #     self.get_data('R1.37', index).astype(np.float32),
        #     self.get_data('R1.61', index).astype(np.float32),
        #     self.get_data('R2.22', index).astype(np.float32),
        #     self.get_data('BT3.72', index).astype(np.float32),  #
        #     self.get_data('BT6.25', index).astype(np.float32),
        #     self.get_data('BT7.10', index).astype(np.float32),
        #     self.get_data('BT8.50', index).astype(np.float32),  #
        #     self.get_data('BT10.8', index).astype(np.float32),  #
        #     self.get_data('BT12.0', index).astype(np.float32),  #
        #     self.get_data('BT13.5', index).astype(np.float32),
        #     self.get_data('DST_binary', index)
        # ])
        multi_data[multi_data == -99.] = np.nan
        if self.transform:
            multi_data = self.transform(torch.from_numpy(multi_data))
        else:
            multi_data = torch.from_numpy(multi_data)
        multi_data[torch.isnan(multi_data)] = 0
        multi_data[torch.isinf(multi_data)] = 0
        multi_image = multi_data[:-1, :, :]
        dsd = multi_data[-1, :, :]
        return multi_image, dsd

    def __len__(self):
        return len(self.get_data_infos('DSD'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    # self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})
                    self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds[:].shape, 'cache_idx': idx})
            # print(file_path)

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path, 'r') as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    # idx = self._add_to_cache(ds.value, file_path)
                    idx = self._add_to_cache(ds[:], file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'],
                               'type': di['type'],
                               'shape': di['shape'],
                               'cache_idx': -1}
                              if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]
