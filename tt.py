# 导入数据增强工具
import Augmentor
import glob
from PIL import Image

# 确定原始图像存储路径以及掩码文件存储路径
# p = Augmentor.Pipeline("./img/lr")
# p.ground_truth("./img/hr")

p = Augmentor.Pipeline("./img/test_lr")
p.ground_truth("./img/test_hr")


# 图像旋转： 按照概率0.8执行，最大左旋角度10，最大右旋角度10
p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)

# 随机翻转
p.flip_random(probability=0.8)
# 图像左右互换： 按照概率0.5执行
p.flip_left_right(probability=0.5)
# 上下翻转
p.flip_top_bottom(probability=0.5)

# 图像放大缩小： 按照概率0.8执行，面积为原始图0.85倍
p.zoom_random(probability=0.3, percentage_area=0.85)
# percentage_area表示裁剪面积占原图像面积的比例，centre指定是否从图片中间裁剪，randomise_percentage_area指定是否随机生成裁剪面积比
p.crop_random(probability=0.3, percentage_area=0.8, randomise_percentage_area=True)

# 最终扩充的数据样本数
p.sample(20)


