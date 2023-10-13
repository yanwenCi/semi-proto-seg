

"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
#from .prostate_2mod import MyDataLoader
from .prostate import MyDataLoader


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, dataroot, split, batch_size, transform=None, catId=2, label_num=None):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        # dataset_class = find_dataset_using_name(opt.dataset_mode)
        # self.dataset = dataset_class(opt)

        if label_num is not None:
            self.dataset_label = MyDataLoader(dataroot, split=split, transform=transform, catId=catId, label_num=label_num)
        else:
            self.dataset_label = MyDataLoader(dataroot, split=split, transform=transform, catId=catId)

        print("labeled dataset [%s] was created " % type(self.dataset_label).__name__)
        self.dataloader_label = torch.utils.data.DataLoader(
            self.dataset_label,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataloader_label)

    def __iter__(self):
        """Return a batch of data"""
        for i, data1 in enumerate(self.dataloader_label):
            if i >= self.__len__():
                print(self.__len__())
                break
            yield data1

