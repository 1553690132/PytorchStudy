from PIL import Image

from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

import numpy as np

img = Image.open('../test_data/train/ants_image/707895295_009cf23188.jpg')

writer = SummaryWriter('logs1')

# tensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image('tensor', img_tensor)

# normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('norm', img_norm)

# resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
writer.add_image('resize', img_resize)

# compose
trans_resize2 = transforms.Resize(212)
trans_compose = transforms.Compose([trans_resize2, trans_tensor])
img_resize2 = trans_compose(img)
writer.add_image('compose', img_resize2)

# randomCrop
trans_random = transforms.RandomCrop(212)
trans_compose_2 = transforms.Compose([trans_random, trans_tensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('random_crop', img_crop, i)

writer.close()
