import torch
from torchvision import transforms, io
from PIL import Image
from torchvision.utils import save_image

from esrgan.read_mrc import read_mrc
import io
#
# # 定义旋转变换
transform = transforms.Compose([
    # transforms.RandomRotation(30), # 随机旋转，最大角度为30度
    transforms.ToTensor() # 转换为tensor
])
#
# img = Image.open('/home/feifei/py/tcan/data/deep/1lr.tif')
# lable = Image.open('/home/feifei/py/tcan/data/deep/1hr.tif')
# img = transform(img)
# lable = transform(lable)
# save_image(img, "tttt.tif", normalize=True)
# save_image(lable, "ssss.tif", normalize=True)


# img = 'data/cell_001/RawSIMData_gt.mrc'
# img = read_mrc(img)[1]
#
# img = transform(img)
#
# print(img.size())
# save_image(img[1], "ssss.tif", normalize=True)


import mrcfile
import matplotlib.pyplot as plt
for i in range(1, 10):
    # 打开MRC文件
    with mrcfile.open('data/cell_001/RawSIMData_level_0%d.mrc' % i, permissive=True) as mrc:
        # 读取图像数据
        image_data = mrc.data

    # 遍历每张图像
    # for i in range(image_data.shape[0]):
    #     # 获取当前图像
    #     single_image = image_data[i, :, :]

    # 显示图像
    plt.imshow(image_data[0], cmap='gray')
    plt.axis('off')
    plt.show()


