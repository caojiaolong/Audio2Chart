"""
this file is used to infer and converts inference result into a playable chart
"""
import torch
import librosa.feature
import json
from model import *

# Create an radom model or load from trained model
# model = ChartNet(20, 20, 200, 512)
# y = model(torch.randn(3, 1, 128, 87))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChartNet(600, 500, 512, 2, 256).to(device)
model.eval()
model.load_state_dict(torch.load('checkpoints/c.pth', map_location=device))


audio_path = "Song.ogg"
# read the audio and compute the mel spectrogram
try:
    y, sr = librosa.load(audio_path)
except BaseException:
    print("audio load fail")

hop_length = 512
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
mel_spectrogram = torch.tensor(mel_spectrogram)
bpm, beat = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units="time")
offset = beat[0] * 1000

# compute the number of possible beats, every beat is divided into 4 sections
audio_length = len(y) / sr + offset / 1000
beats_num = int(audio_length * bpm / 60)
beat_num = beats_num * 4 + 4
period = 60 / (bpm * 4)

# convert mel spectrogram into data_temp
# copied from preprocessing.py
data_temp = {"input": torch.zeros([beat_num, 1, 128, 87],device=device)}
for i in range(beat_num):
    time_now = offset / 1000 + period * i
    frame_now = librosa.time_to_frames(times=time_now, sr=sr, hop_length=hop_length)
    if frame_now - 43 < 0:
        data_temp["input"][i][0] = torch.cat(
            (torch.zeros([128, 43 - frame_now]), mel_spectrogram[:, 0: frame_now + 43 + 1]), dim=1)
    elif frame_now + 43 > mel_spectrogram.shape[1] - 1:
        data_temp["input"][i][0] = torch.cat(
            (mel_spectrogram[:, frame_now - 43:],
             torch.zeros([128, 87 - mel_spectrogram[:, frame_now - 43:].shape[1]])), dim=1)
    else:
        data_temp["input"][i][0] = mel_spectrogram[:, frame_now - 43: frame_now + 43 + 1]

# torch.set_printoptions(profile="full")
# print(data_temp["input"].shape)
outputs = model(data_temp["input"])
outputs = torch.argmax(outputs, dim=1)



# define the chart
chart = {}
chart["meta"] = {"version": "4K ADVANCED Lv.19", "mode": 0, "mode_ext": {"column": 4}}
chart["time"] = [{"beat": [0, 0, 1], "bpm": bpm}]
chart["note"] = []

chart_temp = torch.zeros(beat_num, 4)
for i in range(beat_num):
    chart_temp[i][0] = int(outputs[i] / 1) % 4
    chart_temp[i][1] = int(outputs[i] / 4) % 4
    chart_temp[i][2] = int(outputs[i] / 16) % 4
    chart_temp[i][3] = int(outputs[i] / 64) % 4


def beatindex2note(index):
    return int(index / 4), index % 4


for i in range(beat_num):
    for col in range(4):
        if chart_temp[i][col] == 1:
            beatindex, section = beatindex2note(i)
            chart["note"].append({"beat": [beatindex, section, 4], "column": col})
        elif chart_temp[i][col] == 2:
            beatindex_s, section_s = beatindex2note(i)
            for j in range(i + 1, beat_num):
                beatindex_e, section_e = beatindex2note(j)
                if chart_temp[j][col] == 3:
                    chart["note"].append({"beat": [beatindex_s, section_s, 4], "endbeat": [
                                         beatindex_e, section_e, 4], "column": col})
                    break
                elif chart_temp[j][col] != 0:
                    break

chart["note"].append({"beat": [0, 0, 1], "sound": audio_path, "vol": 100, "offset": offset, "type": 1})
chart["extra"] = {"test": {"divide": 4, "speed": 100, "save": 0, "lock": 0, "edit_mode": 0}}

with open("AIchart.mc", "w", encoding="utf-8") as f:
    json.dump(chart, f)
print(f"generating successfully! ")
