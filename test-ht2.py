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
from models.models import PatchSampleF, BoundaryGenerator,BoundaryGenerator2

predictors = []
attrs = ['Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
         'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
         'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
         'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
         'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
         'Wearing_Necklace', 'Wearing_Necktie', 'Young']
# attrs = ['Pale_Skin']
# # attrs=['Gray_Hair']
# for i in range(len(attrs)):
#     predictor = resnet50(2).eval().cuda()
#     predictor_ckpt = './checkpoints/resnet_predictor/%s_20.pt' % attrs[i]
#     predictor.load_state_dict(torch.load(predictor_ckpt))
#     predictors.append(predictor)
use_resnet=True
attrs=['Smiling','Goatee']
attrs=[]
# attrs=['Gray_Hair','Pale_Skin']
if use_resnet:
    for i in range(len(attrs)):
        predictor = resnet50(2).eval().cuda()
        if attrs[i]=='Old':
            predictor_ckpt = './checkpoints/resnet_predictor/%s_20.pt' % 'Young'
        else:
            predictor_ckpt = './checkpoints/resnet_predictor/%s_20.pt' % attrs[i]
        predictor.load_state_dict(torch.load(predictor_ckpt))
        predictor.eval()
        predictors.append(predictor)

def test(attr,
         num=100,
         img_size=256,
         stylegan_ckpt='checkpoints/ffhq/stylegan2_ffhq.pt',
         save_img=False,
         one_D_for_all=False
         ):
    attrs=attr.split('-')
    predictors = []
    # attrs = ['Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
    #          'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
    #          'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
    #          'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
    #          'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    #          'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    # attrs = ['Pale_Skin']

    l = 8
    save_dir = 'test-results/len=%d/%s' % (l,attr)
    if save_img:
        os.makedirs(save_dir, exist_ok=True)
    stylegan = Generator(img_size, 512, 8).eval().cuda()
    stylegan.load_state_dict(torch.load(stylegan_ckpt)['g_ema'])
    trunc = stylegan.mean_latent(4096).detach()
    latents = torch.load('test-results/latent.pt').cuda()
    G2 = BoundaryGenerator2(fix_len=l).cuda()
    G2.load_state_dict(torch.load('checkpoints/ffhq/%s/010000.pt' % attr)["G"])
    #Gn.load_state_dict(torch.load('checkpoints/ffhq/%s/020000.pt' % attr)["G"])
    @torch.no_grad()
    def generate_img(latent,class_id, len=None):
        label = torch.ones(latent.size(0)).int()*class_id
        latent = stylegan.style(latent)
        syn_latent_edited = G2(latent,label,length0 = len)
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
        logits, probas0 = predictors[i](nn.Upsample(128)(img0))
        logits, probas1 = predictors[i](nn.Upsample(128)(img1))
        return probas1[:, 0] - probas0[:, 0]

    for index,attri in enumerate(attrs[0:1]):
        save_dir = 'test-results/len=%d/%s/%s' % (l,attr,attri)
        os.makedirs(save_dir, exist_ok=True)
        imgs = []
        bs = 8
        cnt = 0
        scores = [0 for i in range(40)]
        num = 50
        resize_512=transforms.Resize([512,512])
        # os.makedirs('samples/ori-512')
        for i in range(num):
            latent = latents[i * bs:i * bs + bs]
            img, img1 = generate_img(latent, index,l)
            img, img2 = generate_img(latent, index, -l)
            #img=resize_512(img)
            for j in range(bs):
                # save_image((img[j] + 1) / 2, os.path.join('samples/ori-512', '%d.jpg' % cnt),
                #            normalize=False)
                if save_img:
                    print(cnt)
                    save_image((torch.stack([img2[j],img[j], img1[j]], 0)+1)/2, os.path.join(save_dir, '%d.jpg' % cnt), normalize=True)
                    #save_image(torch.stack([img2[j], img[j], img1[j]], 0),os.path.join(save_dir, '%d.jpg' % cnt), normalize=True)
                cnt += 1


def test_ablation(attr,
         num=100,
         img_size=256,
         stylegan_ckpt='checkpoints/ffhq/stylegan2_ffhq.pt',
         add_name='',
         save_img=False
         ):
    l = 5
    save_dir = 'test-ablation-results/len=%d/%s' % (l, attr) + add_name
    if save_img:
        os.makedirs(save_dir, exist_ok=True)

    stylegan = Generator(img_size, 512, 8).eval().cuda()
    stylegan.load_state_dict(torch.load(stylegan_ckpt)['g_ema'])
    trunc = stylegan.mean_latent(4096).detach()
    latents = torch.load('test-results/latent.pt').cuda()
    G2 = BoundaryGenerator2(fix_len=l).cuda()
    G2.load_state_dict(torch.load('checkpoints/ablation/%s/020000%s.pt' % (attr, add_name))["G"])

    def generate_img(latent,class_id, len):
        label = torch.ones(latent.size(0))*class_id
        latent = stylegan.style(latent)
        syn_latent_edited = G2(latent,label,length0 = len)
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
        logits, probas0 = predictors[i](nn.Upsample(128)(img0))
        logits, probas1 = predictors[i](nn.Upsample(128)(img1))
        return probas1[:, 0] - probas0[:, 0]

    bs = 8
    cnt = 0
    scores = [0 for i in range(len(attrs))]
    num = 50
    for i in range(num):
        latent = latents[i * bs:i * bs + bs]
        img, img1 = generate_img(latent,0, l)
        img, img2 = generate_img(latent,0, -l)
        if use_resnet:
            for j in range(len(attrs)):
                pros = predict(img, img1, j)
                scores[j] += pros.sum()
        for j in range(bs):
            if save_img:
                save_image((torch.stack([img2[j], img[j], img1[j]], 0) + 1) / 2, os.path.join(save_dir, '%d.jpg' % cnt),
                           normalize=False)
            cnt += 1
    if use_resnet:
        for i in range(len(attrs)):
            print(attrs[i], -scores[i] / cnt)

    bs = 8
    cnt = 0
    scores = [0 for i in range(len(attrs))]
    num = 50
    for i in range(num):
        latent = latents[i * bs:i * bs + bs]
        img, img1 = generate_img(latent,1, l)
        img, img2 = generate_img(latent,1, -l)
        if use_resnet:
            for j in range(len(attrs)):
                pros = predict(img, img1, j)
                scores[j] += pros.sum()
        for j in range(bs):
            if save_img:
                save_image((torch.stack([img2[j], img[j], img1[j]], 0) + 1) / 2, os.path.join(save_dir, '%d.jpg' % cnt),
                           normalize=False)
            cnt += 1
    if use_resnet:
        for i in range(len(attrs)):
            print(attrs[i], -scores[i] / cnt)



def gen():
    latent = torch.randn(5000, 512)
    torch.save(latent, 'test-results/latent-5000.pt')


if __name__ == '__main__':
    # attrs=['Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    # attrs = ['Male']
    #gen()
    #test_attrs = ['Bags_Under_Eyes-Mouth_Slightly_Open','Sideburns-Narrow_Eyes']
    #test_attrs=['Heavy_Makeup-Female','Wearing_Earrings-Gray_Hair']
    test_attrs=['Bangs-Goatee']
    #test_attrs = ['shortwored-simplewshort1']
    for attr in test_attrs:
        print(attr)
        test(attr,save_img=True)
        # test(attr, len=3, stylegan_ckpt='checkpoints/ffhq/danbooru.pt')
        # test(attr, len=5, stylegan_ckpt='checkpoints/ffhq/danbooru.pt')
        #test(attr, len=10, stylegan_ckpt='checkpoints/ffhq/church.pt')
        # test(attr, len=9, stylegan_ckpt='checkpoints/ffhq/danbooru.pt')

    # train('Young')
    # train('Bangs')
    # train('Male')
    # train('Smiling')
    # train('Mouth_Slightly_Open')
    # train('Big_Lips')
    # train('Big_Nose')
    # train(attr_pairs)
    # test('Bangs')