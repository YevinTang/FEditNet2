import os
import argparse
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
from models.models import PatchSampleF, BoundaryGenerator
predictors=[]

def test(attr,
         num=100,
         img_size=256,
         stylegan_ckpt='checkpoints/ffhq/stylegan2_ffhq.pt',
         ckpt=10000):
    save_dir='test-results/len=7/%s/%d'%(attr, ckpt)
    os.makedirs(save_dir,exist_ok=True)


    stylegan = Generator(img_size, 512, 8).eval().cuda()
    stylegan.load_state_dict(torch.load(stylegan_ckpt)['g_ema'])
    trunc = stylegan.mean_latent(4096).detach()
    latents=torch.load('test-results/latent.pt').cuda()
    G2 = BoundaryGenerator(fix_len=5).cuda()#5-15,尽量往大的去
    G2.load_state_dict(torch.load('checkpoints/ffhq/%s/%06d.pt'%(attr, ckpt))["G"])
    @torch.no_grad()
    def generate_img(latent, len):
        latent = stylegan.style(latent)
        syn_latent_edited = G2(latent, length = len)
        img0, _, _ = stylegan(
            [latent],
            truncation=0.7,
            truncation_latent=trunc,
            input_is_latent=True,
            randomize_noise=False,
        )
        img1, _, _ = stylegan(
            [syn_latent_edited],
            truncation=0.7,
            truncation_latent=trunc,
            input_is_latent=True,
            randomize_noise=False,
        )
        return img0, img1

    
    @torch.no_grad()
    def predict(img0,img1,i):
        logits, probas0 = predictors[i](nn.Upsample(128)(img0))
        logits, probas1 = predictors[i](nn.Upsample(128)(img1))
        #pred = torch.argmax(probas0, dim=1)
        # print(pred)
        # print('0',probas0[:,0])
        # print('1',probas1[:,0])
        return probas1[:,0]-probas0[:,0]

    imgs = []
    preds = []
    bs=8
    cnt=0
    scores=[0 for i in range(40)]
    num=30
    for i in range(num):
        print(i)
        latent=latents[i*bs:i*bs+bs]
        img, img1 = generate_img(latent, 7)
        img, img2 = generate_img(latent, -7)

        imgs.append(img)
        for j in range(bs):
            save_image((torch.stack([img2[j],img[j],img1[j]],0)+1)/2, os.path.join(save_dir,'%d.jpg'%cnt), normalize=False)
            cnt+=1
    svg_num=num*bs
    # with open('tmp.txt','a')as f:
    #     for i in range(len(attrs)):
    #         f.write(attrs[i][:5]+' %.4f\n'%float(scores[i]/svg_num))


def gen():
    latent =torch.randn(400, 512)
    l0=torch.load('test-results/latent.pt')
    latent=torch.cat([l0,latent],dim=0)
    torch.save(latent,'test-results/latent2.pt')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr', type=str, default='checkpoints_smiles')
    attrs=[parser.parse_args().attr]
    for attr in attrs:
        print(attr)
        test(attr=attr, ckpt=30000)
