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
from models.models import PatchSampleF, BoundaryGenerator

predictors = []
# attrs = ['Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
#          'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
#          'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
#          'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
#          'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
#          'Wearing_Necklace', 'Wearing_Necktie', 'Young']
# attrs=['Rosy_Cheeks']
# # # attrs=['Gray_Hair']
# for i in range(len(attrs)):
#     predictor = resnet50(2).eval().cuda()
#     predictor_ckpt = './checkpoints/resnet_predictor/%s_20.pt' % attrs[i]
#     predictor.load_state_dict(torch.load(predictor_ckpt))
#     predictors.append(predictor)


def test(attr,
         num=100,
         img_size=256,
         stylegan_ckpt='checkpoints/ffhq/stylegan2_ffhq.pt',
         ):
    save_dir = 'test-results/%s' % attr
    os.makedirs(save_dir, exist_ok=True)
    attr1,attr2=attr.split('-')
    stylegan = Generator(img_size, 512, 8).eval().cuda()
    stylegan.load_state_dict(torch.load(stylegan_ckpt)['g_ema'])
    trunc = stylegan.mean_latent(4096).detach()
    latents = torch.load('test-results/latent.pt').cuda()
    G1 = BoundaryGenerator(fix_len=7).cuda()
    G2 = BoundaryGenerator(fix_len=7).cuda()
    dict1=torch.load('checkpoints/ffhq/%s/010000.pt' % attr1)["G"]
    dict2=torch.load('checkpoints/ffhq/%s/010000.pt' % attr2)["G"]
    w1=dict1['boundary']
    w2=dict2['boundary']
    new_w2=w2-((w1*w2).sum()/(w1*w1).sum())*w1
    new_w1 = w1 - ((w1 * w2).sum() / (w2 * w2).sum()) * w2
    dict1['boundary']=new_w1
    dict2['boundary']=new_w2
    G1.load_state_dict(dict1)
    G2.load_state_dict(dict2)
    @torch.no_grad()
    def generate_img(latent,class_id,l):
        latent = stylegan.style(latent)
        if class_id == 0:
            syn_latent_edited = G1(latent,l)
        if class_id==1:
            syn_latent_edited = G2(latent,l)
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
    def predict(img0, img1, i):
        print(img0.min(), img1.max())
        logits, probas0 = predictors[i](nn.Upsample(128)(img0))
        logits, probas1 = predictors[i](nn.Upsample(128)(img1))
        # pred = torch.argmax(probas0, dim=1)
        # print(pred)
        # print('0',probas0[:,0])
        # print('1',probas1[:,0])
        return probas1[:, 0] - probas0[:, 0]

    save_dir = 'test-results/%s/%s-vertical' % (attr, attr1)
    os.makedirs(save_dir, exist_ok=True)
    imgs = []
    bs = 8
    cnt = 0
    scores = [0 for i in range(40)]
    num = 30
    l=3
    for i in range(num):
        print(i)
        latent = latents[i * bs:i * bs + bs]
        img, img1 = generate_img(latent, 0,l)
        img, img2 = generate_img(latent, 0,-2)
        # for j in range(len(attrs)):
        #     pros = predict(img, img1, j)
        #     scores[j] = pros.sum()
        imgs.append(img)
        for j in range(bs):
            save_image(torch.stack([img2[j],img[j], img1[j]], 0), os.path.join(save_dir, '%d.jpg' % cnt), normalize=True,
                       range=(-1, 1))
            cnt += 1

    save_dir = 'test-results/%s/%s-vertical' % (attr, attr2)
    os.makedirs(save_dir, exist_ok=True)
    imgs = []
    bs = 8
    cnt = 0
    scores = [0 for i in range(40)]
    num = 30
    for i in range(num):
        print(i)
        latent = latents[i * bs:i * bs + bs]
        img, img1 = generate_img(latent, 1,l)
        img, img2 = generate_img(latent, 1,-l)
        # for j in range(len(attrs)):
        #     pros = predict(img, img1, j)
        #     scores[j] = pros.sum()
        imgs.append(img)
        for j in range(bs):
            save_image(torch.stack([img2[j],img[j], img1[j]], 0), os.path.join(save_dir, '%d.jpg' % cnt), normalize=True,
                       range=(-1, 1))
            cnt += 1
    # with open('tmp.txt','a')as f:
    #     for i in range(len(attrs)):
    #         f.write(attrs[i][:5]+' %.4f\n'%float(scores[i]/svg_num))


def gen():
    latent = torch.randn(100, 512)
    torch.save(latent, 'test-results/latent.pt')


if __name__ == '__main__':
    # attrs=['Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    # attrs = ['Male']
    # gen()
    test_attrs = ['Bangs-Goatee']
    for attr in test_attrs:
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