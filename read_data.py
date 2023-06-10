from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import glob

class ImageDataset(Dataset):

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.transform2 = transforms.Compose([transforms.ToTensor()])
        self.lr_imgs = sorted(glob.glob(root + "/*lr.*"))
        self.hr_imgs = sorted(glob.glob(root + "/*hr.*"))

    def __getitem__(self, idx):
        lr_img = self.lr_imgs[idx]
        lr_img = Image.open(lr_img)
        lr_img = self.transform(lr_img)

        hr_img = self.hr_imgs[idx]
        hr_img = Image.open(hr_img)

        hr_img = self.transform2(hr_img)
        sample = {'lr': lr_img, 'hr': hr_img}
        return sample

    def __len__(self):
        assert len(self.lr_imgs) == len(self.hr_imgs)
        return len(self.lr_imgs)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    train_dataset = ImageDataset("data/deep", transform)

    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0)


    for i, j in enumerate(dataloader):

        # imgs, labels = j
        print(i, j['lr'].shape)
        # writer.add_image("train_data_b2", make_grid(j['img']), i)




