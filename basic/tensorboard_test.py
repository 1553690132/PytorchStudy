from torch.utils.tensorboard import SummaryWriter

import numpy as np

from PIL import Image

writer = SummaryWriter('../logs')
img_path = '../test_data/train/ants_image/175998972.jpg'
img_PIL = Image.open(img_path)

img_array = np.array(img_PIL)

writer.add_image('test', img_array, 1, dataformats='HWC')

img_PIL2 = Image.open('../test_data/train/ants_image/0013035.jpg')

writer.add_image('test', np.array(img_PIL2), 2, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=2*x", 2 * i, i)

writer.close()
