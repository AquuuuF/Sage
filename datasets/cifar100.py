from collections import defaultdict
import numpy as np
import torchvision
from PIL import Image

class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_factor=None, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

        if train and imb_factor is not None:
            np.random.seed(rand_number)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_factor)
            self.gen_imbalanced_data(img_num_list)

        self.classnames = self.classes
        self.labels = self.targets
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        
    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)### close it when conduct visualizations !!!
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list
    

    ## for proco
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        
        if type(self.transform) == list:
            if self.transform is not None:
                if self.train:
                    sample1 = self.transform[0](img)
                    sample2 = self.transform[1](img)
                    sample3 = self.transform[2](img) 
                    # print('list')
                    return [sample1, sample2, sample3], label
                else:
                    return self.transform(img), label
        else:
            return self.transform(img), label

        if self.target_transform is not None:
            label = self.target_transform(label)



class CIFAR100(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=None, train=train, transform=transform)


class CIFAR100_IR10(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, train=train, transform=transform)


class CIFAR100_IR50(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.02, train=train, transform=transform)


class CIFAR100_IR100(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, train=train, transform=transform)
