import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid


# images = (glob.glob("img/lr/output/*"))
images = (glob.glob("img/test_lr/output/*"))

for i, path in enumerate(images):
    img = images[i]
    img = Image.open(img)

    t = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    img = t(img)


    # img = img.resize((128, 128))



    # # 文件名
    # file = path.split('_')[-1].split('.')[0]
    # if '_groundtruth_' in path:
    #     file += '_hr'
    # else:
    #     file += '_lr'
    # 保存PIL图像
    # img.save("data/deep2/%s.tif" % file)

    # 文件名
    file = path.split('_')[-1].split('.')[0]
    if '_groundtruth_' in path:
        file += '_hr'
    else:
        file += '_lr'
    # save_image(img, "data/deep2/%s.tif" % file, normalize=True)
    save_image(img, "data/test/%s.tif" % file, normalize=True)


