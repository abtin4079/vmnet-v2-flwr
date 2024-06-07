import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from kvasir import transforms as T
# Additional Scripts
from torch.utils.data import DataLoader, random_split

class DentalDataset(Dataset):
    #output_size = cfg.transunet.img_dim

    def __init__(self, path, transform, sail_path, output_size):
        super().__init__()

        self.transform = transform
        self.output_size = output_size
        img_folder = os.path.join(path, 'img')
        img_sail_folder = os.path.join(sail_path, 'grad_img')
        mask_folder = os.path.join(sail_path, 'mask')

        self.img_paths = []
        self.img_sail_paths = []
        self.mask_paths = []
        for p in os.listdir(img_sail_folder):
            name = p.split('.')[0]

            self.img_paths.append(os.path.join(img_folder, name + '.jpg'))
            self.img_sail_paths.append(os.path.join(img_sail_folder, name + '.png'))
            self.mask_paths.append(os.path.join(mask_folder, name + '.jpg'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_paths[idx]
        img_sail = self.img_sail_paths[idx]
        mask = self.mask_paths[idx]

        img_sail = cv2.imread(img_sail)
        img_sail = cv2.cvtColor(img_sail, cv2.COLOR_BGR2RGB)
        img_sail = cv2.resize(img_sail, (self.output_size, self.output_size))


        # preprocessing of original images
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.output_size, self.output_size))
        

        # preprocessing of masks
        mask = cv2.imread(mask, 0)
        mask = cv2.resize(mask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=-1)

        sample = {'img': img, 'img_sail': img_sail, 'mask': mask}
        

        #print("Before transform:", sample)
        if self.transform:
            sample = self.transform(sample)
        #print("After transform:", sample)

        
        img, img_sail, mask = sample['img'], sample['img_sail'], sample['mask']

        img = img / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.astype('float32'))

        img_sail = img_sail / 255.
        img_sail = img_sail.transpose((2, 0, 1))
        img_sail = torch.from_numpy(img_sail.astype('float32'))
        
        mask = mask / 255.
        mask = mask.transpose((2, 0, 1))
        mask = torch.from_numpy(mask.astype('float32'))

        return {'img': img, 'img_sail' : img_sail, 'mask': mask}

    def __len__(self):
      return len(self.img_paths)
    

def prepare_dataset_kvasir(num_partitions: int, batch_size: int, output_size: int, val_ratio: float = 0.1):
    """Download MNIST and generate IID partitions."""

    # download MNIST in case it's not already in the system

    # trainset, testset = get_mnist()

    # set the custom transform
    transform = transforms.Compose([T.RandomAugmentation(2)])

    trainset = DentalDataset('/content/drive/MyDrive/kvasir/train',
                       transform, '/content/drive/MyDrive/kvasir/train', output_size=output_size)
    
    testset = DentalDataset('/content/drive/MyDrive/kvasir/test',
                       transform, '/content/drive/MyDrive/kvasir/test', output_size=output_size)




    num_images = len(trainset) // num_partitions

    # a list of partition lenghts (all partitions are of equal size)
    partition_len = [num_images] * num_partitions
    print(len(partition_len))
    print(num_partitions)
    trainset_length = len(trainset)
    print("Trainset length:", trainset_length)

    # split randomly. This returns a list of trainsets, each with `num_images` training examples
    # Note this is the simplest way of splitting this dataset. A more realistic (but more challenging) partitioning
    # would induce heterogeneity in the partitions in the form of for example: each client getting a different
    # amount of training examples, each client having a different distribution over the labels (maybe even some
    # clients not having a single training example for certain classes). If you are curious, you can check online
    # for Dirichlet (LDA) or pathological dataset partitioning in FL. A place to start is: https://arxiv.org/abs/1909.06335
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    # for each train set, let's put aside some training examples for validation
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # construct data loaders and append to their respective list.
        # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    # We leave the test set intact (i.e. we don't partition it)
    # This test set will be left on the server side and we'll be used to evaluate the
    # performance of the global model after each round.
    # Please note that a more realistic setting would instead use a validation set on the server for
    # this purpose and only use the testset after the final round.
    # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
    # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
    # in main.py above the strategy definition for more details on this)
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader




if __name__ == '__main__':
    import torchvision.transforms as transforms
    import kvasir.transforms as T
    transform = transforms.Compose([T.RandomAugmentation(2)])

    md = DentalDataset('F:/UNIVERCITY/sharifian/t1/datasets/tumor_dataset/process/train',
                       transform, 'F:/UNIVERCITY/sharifian/t1/datasets/tumor_dataset/sailency/train')
    loader = DataLoader(md, batch_size=cfg.batch_size, shuffle=True)

    print(loader)
    # for sample in md:
    #     print(sample['img'].shape)
    #     print(sample['mask'].shape)
    #     print(sample['img_sail'].shape)

    #     break
