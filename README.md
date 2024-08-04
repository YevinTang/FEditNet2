# FEditNet++
[TPAMI] FEditNet: Few-shot Editing of Latent Semantics in GAN Spaces

###  [Paper](https://ieeexplore.ieee.org/document/10607942) | [Suppl](10.1109/TPAMI.2024.3432529/mm1)

<!-- <br> -->
[Ran Yi](https://yiranran.github.io/),
[Teng Hu](https://github.com/sjtuplayer),
[Mengfei Xia](https://github.com/thuxmf), 
[Yizhe Tang](https://github.com/YevinTang),
 and [Yongjin Liu](https://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm),
<!-- <br> -->

![image](imgs/framework.png)


# Prepare

```bash
#conda create -n live python=3.7
#conda activate live
#conda install -y pytorch torchvision -c pytorch
#conda install -y numpy scikit-image
#conda install -y -c anaconda cmake
#conda install -y -c conda-forge ffmpeg
#pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom
#pip install opencv-python==4.5.4.60  # please install this version to avoid segmentation fault.
#
#cd DiffVG
#git submodule update --init --recursive
#python setup.py install
#cd ..
```



## Training Step

### (0) Prepare
Data prepare: Download the [StyleGAN](https://image-net.org) checkpoint.

### (1) Train the editnet Model

Put the downloaded Imagenet or any dataset you want into `$path_to_the_dataset`. 
Then, you can train the coarse-stage model by running:

```
python3 train_editnet.py --name=$attribute_name
```

After training, the checkpoints and logs are saved in the directory ``.

### (2) Train the editnet2 Model

Coming soon

[//]: # (With the trained coarse-stage model, you can train the refinement-stage model by running:)

[//]: # ()
[//]: # (```)

[//]: # (python3 main_refine --data_path=$path_to_the_dataset)

[//]: # (```)

[//]: # ()
[//]: # (After training, the checkpoints and logs are saved in the directory `output_refine`.)

## Citation

If you find this code helpful for your research, please cite:

```

```
