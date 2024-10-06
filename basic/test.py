from torch.utils.tensorboard import SummaryWriter

from PIL import Image

from torchvision import transforms

writer = SummaryWriter('../logs')

img = Image.open('../test_data/train/ants_image/201558278_fe4caecc76.jpg')
img_tensor = transforms.ToTensor()(img)

writer.add_image('tensor', img_tensor)

img_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
writer.add_image('norm', img_norm)

img_resize = transforms.Resize((224, 224))(img_tensor)
writer.add_image('resize', img_resize)

for i in range(10):
    img_crop = transforms.RandomCrop(224)(img_tensor)
    writer.add_image('crop', img_crop, i)

img_compose = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(img)

writer.add_image('compose', img_compose)

writer.close()
