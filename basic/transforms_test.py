from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

from PIL import Image

# tensor 数据类型
img_path = '../test_data/train/ants_image/175998972.jpg'
img_PIL = Image.open(img_path)

tensor_trans = transforms.ToTensor()

tensor_img = tensor_trans(img_PIL)
print(tensor_img)

writer = SummaryWriter('../logs')
writer.add_image('transforms', tensor_img)

writer.close()
