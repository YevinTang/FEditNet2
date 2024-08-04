import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torch.optim import Adam
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models.stylegan_model import Generator
from models.models import resnet50
from data.celeba_attrimg_dataset import AttrImgDataset
from torchvision.utils import save_image
from PIL import Image

def test(attr='Male', epochs=20, save_freq=2):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dir='dataset/celeba-test/Gray_Hair'
    file_names=os.listdir(dir)
    file_paths=[os.path.join(dir,i) for i in file_names]
    imgs=[]
    for file_path in file_paths:
        img = transform(Image.open(file_path))
        imgs.append(img)
    imgs=torch.stack(imgs).cuda()[:20]
    label=torch.ones(imgs.size(0)).cuda().int()
    model = resnet50(2).cuda().eval()
    #print(attr)
    predictor_ckpt = './checkpoints/resnet_predictor/%s_20.pt' % attr
    model.load_state_dict(torch.load(predictor_ckpt))
    crit = nn.CrossEntropyLoss()
    #print(imgs.min(),imgs.max(),imgs.shape)
    logits, prob = model(imgs)
    #loss = torch.nn.functional.cross_entropy(logits, label)
    tot = label.numel()
    acc = (torch.argmax(prob, dim=1) == label).sum().item()
    #print(prob)
    #print(loss)
    print(acc/len(imgs))


if __name__ == '__main__':
    attr_pairs = [
        'Young',
        'Bangs',
        'Male',
        'Smiling',
    ]
    attrs = ['Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
             'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
             'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    # attrs = ['Pale_Skin']
    #attrs=['Gray_Hair']
    attrs = ['Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
              'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
              'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
              'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
              'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
              'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    #attrs = ['Bald']
    for attr in attrs:
        print(attr)
        test(attr)

    # train('Young')
    # train('Bangs')
    # train('Male')
    # train('Smiling')
    # train('Mouth_Slightly_Open')
    # train('Big_Lips')
    # train('Big_Nose')
    # train(attr_pairs)
    # test('Bangs')