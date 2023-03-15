# Audio2Chart

## Description
This repository is used to restore my final homework in AI class, I tried to implement a neural network for converting any audio into a playable 4 Keys Malody chart (Malody is a rhythm action game).

## Network Architecture

The overall architecture of the ann consists of two nets:

1. CNN for analysing audio feature
2. Bi-LSTM for generating charts

## Usage

### For preprocessing data: 

Put all of the chart file like `***.mcz` in a folder named "data_raw", and run `python unzip.py`, then run `python datafilter.py`, we will get two folders named `data_unziped` and `data`. Above all aims to get the train data and corresponding labels. At last, run `python preprocessing.py` to get a file named `data.pkl` to save matrixing data. 

### For training model from scratch or continuing training: 

Just modify `train.py` and change the path of pretrained model. 

### For inference:

Run `python infer.py audio_path [-m model_path]`

The generated chart will be at the root of this file

## File Structure

`checkpoints`: for saving trained model

`data_raw`: original data, download from [malody_official_chart](https://cbo17ty22x.feishu.cn/wiki/wikcncFuigGA1V7C9ffxcnWHSvd)

`data`, `data_unzip`, `data.pkl`: for preprocessing data

`model.py`: model file

`train.py`: train the model

`infer.py`: use trained model to infer

## Reference:

1. Donahue, Chris, Zachary C. Lipton, and Julian McAuley. “Dance Dance Convolution.” arXiv, June 20, 2017. http://arxiv.org/abs/1703.06891.
2. Takada, Atsushi, Daichi Yamazaki, Likun Liu, Yudai Yoshida, Nyamkhuu Ganbat, Takayuki Shimotomai, Taiga Yamamoto, Daisuke Sakurai, and Naoki Hamada. “Gen\’eLive! Generating Rhythm Actions in Love Live!” arXiv, December 20, 2022. https://doi.org/10.48550/arXiv.2202.12823.

also inspired by https://github.com/nladuo/AI_beatmap_generator