import argparse
import os
import numpy as np
import math
import pytorch_ssim

from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


from models.gan import *
from tcan.datasets import ImageDataset
from torchvision import transforms


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval betwen image samples")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--dataset_name", type=str, default="deep", help="name of the dataset")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")

opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
hr_shape = (opt.hr_height, opt.hr_width)


cuda = True if torch.cuda.is_available() else False


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator(input_shape=img_shape)
# Losses
mse_loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()
ssim_loss = pytorch_ssim.SSIM(window_size=11)
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    bce_loss = bce_loss.cuda()
    mse_loss = mse_loss.cuda()
    ssim_loss = ssim_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
dataloader = DataLoader(
    ImageDataset("data/%s" % opt.dataset_name, transforms.Compose([transforms.Resize(128), transforms.ToTensor()])),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
    )

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):


        imgs_lr = imgs["lr"].cuda()
        real_imgs = imgs["hr"].cuda()

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        gen_imgs = generator(imgs_lr)
        print(imgs_lr.size())

        transform = transforms.Compose([transforms.Resize((256, 256))])

        gen_imgs = transform(gen_imgs)
        d_hr_imgs = discriminator(gen_imgs)
        d_real_imgs = discriminator(real_imgs)

        valid = Variable(Tensor(gen_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(gen_imgs.size(0), 1).fill_(0.0), requires_grad=False)


        g_loss = mse_loss(gen_imgs,real_imgs)
        g_loss += ssim_loss(gen_imgs,real_imgs)
        g_loss += bce_loss(d_hr_imgs - d_real_imgs, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        d_loss = 0.5 * (bce_loss(d_hr_imgs.detach() - d_real_imgs.detach(), fake) + bce_loss(d_real_imgs.detach() - d_hr_imgs.detach(), valid))
        d_loss.requires_grad_(True)
        d_loss.backward()

        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=5, normalize=True)
            # img_grid = torch.cat((imgs_lr, gen_imgs), -1)
            save_image(imgs_lr, "images/%dl.png" % batches_done, normalize=False)
            save_image(gen_imgs, "images/%dh.png" % batches_done, nrow=5, normalize=True)
