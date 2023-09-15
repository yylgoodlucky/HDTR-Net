# HDTR: A Real-Time High-Definition Teeth Restoration Network for Arbitrary Talking Face Generation Methods

We propose `A Real-Time High-Definition Teeth Restoration Network (HDTR-Net)` to address talking face videos with blurred mouth 
in this work, which aims to improve clarity of talking face mouth and lip regions in real-time inference 
that correspond to given arbitrary talking face videos.

[[Paper]](https://arxiv.org/abs/2309.07495)

<img src='./docs/img/HDTR-Net.png' width=880>

### Recommondation of our works
This repo is maintaining by authors, if you have any questions, please contact us at issue tracker.

**The official repository with Pytorch**
**Our method can restorate teeth region for arbitrary face generation on images and videos**

## Test Results
![Results1](./docs/img/male.png)
![Results2](./docs/img/male_HQ.png)
![Results3](./docs/img/female_side-face.png)
![Results4](./docs/img/female_HQ_side-face.png)

## Requirements
* [python](https://www.python.org/download/releases/)（We use version 3.7)
* [PyTorch](https://pytorch.org/)（We use version 1.13.1)
* [opencv2](https://opencv.org/releases.html)
* [ffmpeg](https://ffmpeg.org/)

We conduct the experiments with 4 32G V100 on CUDA 10.2. For more details, please refer to the `requirements.txt`. We recommend to install [pytorch](https://pytorch.org/) firstly, and then run:
```
pip install -r requirements.txt
```

## Generating test results
* Download the pre-trained model [checkpoint](https://drive.google.com/drive/folders/1IGJpQVC2fbJJASoS7bbPdt722vSvMtHr?hl=zh-cn) 
Create the default folder `./checkpoint` and put the checkpoint in it or get the CHECKPOINT_PATH, Then run the following 

bash
``` 
CUDA_VISIBLE_DEVICES=0 python inference.py
```
To inference on other videos, please specify the `--input_video` option and see more details in code.


## Citation and Star
Please cite the following paper and star this project if you use this repository in your research. Thank you!
```
@misc{li2023hdtrnet,
      title={HDTR-Net: A Real-Time High-Definition Teeth Restoration Network for Arbitrary Talking Face Generation Methods}, 
      author={Yongyuan Li and Xiuyuan Qin and Chao Liang and Mingqiang Wei},
      year={2023},
      eprint={2309.07495},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
},
```