# Audio2Chart

## Description
this repository is used for restore my final homework in AI class, I tried to implement a ANN for converting any audio into a playable rhythm action games chart.

## Network Architecture

the overall architecture of the ann consists of two nets:

1. CNN for analysing audio feature
2. Bi-LSTM for generating charts

## Usage

## File

`model.py`: model file

`train.py`: train the model

`infer.py`: use trained model to infer

## Reference:

1. Donahue, Chris, Zachary C. Lipton, and Julian McAuley. “Dance Dance Convolution.” arXiv, June 20, 2017. http://arxiv.org/abs/1703.06891.
2. Takada, Atsushi, Daichi Yamazaki, Likun Liu, Yudai Yoshida, Nyamkhuu Ganbat, Takayuki Shimotomai, Taiga Yamamoto, Daisuke Sakurai, and Naoki Hamada. “Gen\’eLive! Generating Rhythm Actions in Love Live!” arXiv, December 20, 2022. https://doi.org/10.48550/arXiv.2202.12823.

also inspired by https://github.com/nladuo/AI_beatmap_generator