![Psyche AI Inc release](./media/psy_logo.png)

# EmoTalk: Speech-Driven Emotional Disentanglement for 3D Face Animation [ICCV2023]

Official PyTorch implementation for the paper:

> **EmoTalk: Speech-Driven Emotional Disentanglement for 3D Face Animation**, ***ICCV 2023***.
>
> Ziqiao Peng, Haoyu Wu, Zhenbo Song, Hao Xu, Xiangyu Zhu, Hongyan Liu, Jun He, Zhaoxin Fan
>
> <a href='https://arxiv.org/abs/2303.11089'><img src='https://img.shields.io/badge/arXiv-2303.11089-red'></a> <a href='https://ziqiaopeng.github.io/emotalk/'><img src='https://img.shields.io/badge/Project-Video-Green'></a> [![License ↗](https://img.shields.io/badge/License-CCBYNC4.0-blue.svg)](LICENSE)


<p align="center">
<img src="./media/emotalk.png" width="90%" />
</p>

> Given audio input expressing different emotions, EmoTalk produces realistic 3D facial animation sequences with corresponding emotional expressions as outputs.

## Environment

- Linux
- Python 3.8.8
- Pytorch 1.12.1
- CUDA 11.3
- Blender 3.4.1
- ffmpeg 4.4.1

Clone the repo:
  ```bash
  git clone https://github.com/psyai-net/EmoTalk_release.git
  cd EmoTalk_release
  ```  
Create conda environment:
```bash
conda create -n emotalk python=3.8.8
conda activate emotalk
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```


## **Demo**
Download Blender and put it in this directory.
```bash
wget https://mirror.freedif.org/blender/release/Blender3.4/blender-3.4.1-linux-x64.tar.xz
tar -xf blender-3.4.1-linux-x64.tar.xz
mv blender-3.4.1-linux-x64 blender && rm blender-3.4.1-linux-x64.tar.xz
```
Download the pretrained models from [EmoTalk.pth](https://drive.google.com/file/d/1gMWRI-w4NJlvWuprvlUUpdkt6Givy_em/view?usp=drive_link) . Put the pretrained models under `pretrain_model` folder. 
Put the audio under `aduio` folder and run
```bash
python demo.py --wav_path "./audio/disgust.wav"
```
The generated animation will be saved in `result` folder.


## **Dataset**
Coming soon...

## **Citation**
If you find this work useful for your research, please cite our paper:
```
  @inproceedings{peng2023emotalk,
    title={EmoTalk: Speech-driven emotional disentanglement for 3D face animation}, 
    author={Ziqiao Peng and Haoyu Wu and Zhenbo Song and Hao Xu and Xiangyu Zhu and Hongyan Liu and Jun He and Zhaoxin Fan},
    journal={arXiv preprint arXiv:2303.11089},
    year={2023}
  }
```

## **Acknowledgement**
Here are some great resources we benefit:
- [Faceformer](https://github.com/EvelynFan/FaceFormer) for training pipeline
- [EVP](https://github.com/jixinya/EVP) for training dataloader
- [Speech-driven-expressions](https://github.com/YoungSeng/Speech-driven-expressions) for rendering
- [Wav2Vec2 Content](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) and [Wav2Vec2 Emotion](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition) for audio encoder
- [Head Template](http://filmicworlds.com/blog/solving-face-scans-for-arkit/) for visualization.

Thanks to John Hable for sharing his head template under the CC0 license, which is very helpful for us to visualize the results.

## **Contact**
For questions, please contact pengziqiao@ruc.edu.cn

For commercial licensing, please contact fanzhaoxin@psyai.net

## **License**
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. Please read the [LICENSE](LICENSE) file for more information.



