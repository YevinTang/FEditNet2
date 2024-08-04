import sys
sys.path.append("..")
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import csv
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import random
import cv2
from Facecrop.FaceBoxes import FaceBoxes

class Discriminator_AU(nn.Module):
    def __init__(self, input_nc=3, aus_nc=17, image_size=128, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d):
        super(Discriminator_AU, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.01, True)
        ]
        cur_dim = ndf
        for n in range(1, n_layers):
            sequence += [
                nn.Conv2d(cur_dim, 2 * cur_dim,
                          kernel_size=kw, stride=2, padding=padw, bias=nn.InstanceNorm2d),
                nn.LeakyReLU(0.01, True)
            ]
            cur_dim = 2 * cur_dim

        self.model = nn.Sequential(*sequence)
        # patch discriminator top
        self.dis_top = nn.Conv2d(cur_dim, 1, kernel_size=kw-1, stride=1, padding=padw, bias=False)
        # AUs classifier top
        k_size = int(image_size / (2 ** n_layers))
        self.aus_top = nn.Conv2d(cur_dim, aus_nc, kernel_size=k_size, stride=1, bias=False)

        # from torchsummary import summary
        # summary(self.model.to("cuda"), (3, 128, 128))

    def forward(self, img):
        if img.shape[1]!=128 or img.shape[2]!=128:
            resize = transforms.Resize([128, 128])
            img=resize(img)
        if img.max()>10 and img.min()>=0:
            img=(img-127.5)/127.5
        if img.min()>=0 and img.max()<2:
            img=img*2-1
        embed_features = self.model(img)
        pred_aus = self.aus_top(embed_features)
        return pred_aus.squeeze()

def crop(y_hat, size=128):
    try:
        face_boxes = FaceBoxes(cuda=True)
        y_crop = None
        resize = transforms.Resize([size, size])
        for img in y_hat:
            det = face_boxes(img[[2, 1, 0]])
            det = list(map(int, det))
            img_crop = img[:, det[1]:det[3], det[0]:det[2]]
            img_crop = resize(img_crop)
            # save_image(img_crop.cpu().detach().float(), "results/%d.png" % random.randint(0,100000), nrow=1, normalize=True)
            img_crop = img_crop.unsqueeze(0)
            if y_crop == None:
                y_crop = img_crop
            else:
                y_crop = torch.cat((y_crop, img_crop), dim=0)
        return y_crop
    except:
        print("error")
        return y_hat


if __name__ == '__main__':
    D = Discriminator_AU().cuda()
    D.load_state_dict(torch.load("/data2/huteng_home/pixel2style2pixel/models/D-AU3.pth"))
    D.eval()
    img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256])
        ])
    criterionMSE = torch.nn.MSELoss()
    f = open("/data2/huteng_home/dataset/rafd/rafd_au.txt")
    lines = f.readlines()
    totalloss = 0
    cnt = 0
    for line in lines:
        data = line.split(" ")
        imgname = data[0] + ".jpg"
        gt = []
        for i in data[1:]:
            gt.append(float(i))
        gt = torch.tensor(gt).cuda()
        img = Image.open("/data2/huteng_home/dataset/rafd/RafD_aligned8/" + imgname).convert('RGB')
        img = img_transform(img).cuda()
        # img = img.unsqueeze(0)
        # img = crop(img)
        # save_image(img, 'test%d.jpg' % cnt, normalize=False)
        # print(img)
        pred = D(img)
        loss = criterionMSE(gt, pred)
        print('loss: %f' % loss)
        totalloss += loss
        cnt += 1
        if cnt == 5:
            break
    print("average loss: %f" % (totalloss / cnt))

