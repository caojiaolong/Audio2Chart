"""
this file is used to:

1. compute mel spectrogram
2. according to bpm and beat, split spectrogram into many 2 seconds pieces
3. save the results above and the label into the shape: {index: {"input": [beat_num, 1, 128, 87],"label": [beat_num, 256]}}

for example: to get a sample from data.json, we can use `data[index]["input"]` and `data[index]["label"]`
"""
import librosa
import librosa.feature
import torch
import json
import os
import pickle


def note2beatindex(beat):
    """
    convert beat like into the index of corresponding section
    :return: (int)
    """
    return int(beat[0] * 4 + 4 / beat[2] * beat[1])


path_data = "./data"
num_data = len(os.listdir(path_data))

data = {}
counter = 0
for index in range(num_data):
    # get the names of audio and chart
    audio = ""
    chart = ""
    for file in os.listdir(path_data + "/" + str(index)):
        if file.endswith(".ogg"):
            audio = file
        elif file.endswith(".mc"):
            chart = file

    # read the audio and compute the mel spectrogram
    try:
        y, sr = librosa.load(path_data + "/" + str(index) + "/" + audio)
    except BaseException:
        print(f"fail, skip index: {index}")
        continue
    hop_length = 512
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    mel_spectrogram = torch.tensor(mel_spectrogram)

    # read the chart file, get bpm and offset
    chart = open(path_data + "/" + str(index) + "/" + chart, "r", encoding="utf-8")
    chart = json.load(chart)
    bpm = chart["time"][0]["bpm"]
    offset = chart["note"][-1].get("offset", 0)

    # get rid of the chart that changes bpm
    if len(chart["time"]) > 1:
        print(f"bpm changes, skip index:{index}")
        continue

    # compute the number of possible beats, every beat is divided into 4 sections
    audio_length = len(y) / sr + offset / 1000
    beats_num = int(audio_length * bpm / 60)
    beat_num = beats_num * 4 + 20
    period = 60 / (bpm * 4)

    # define the temporary data, which will be embedded into data later
    # convert mel spectrogram into data_temp
    data_temp = {"input": torch.zeros([beat_num, 1, 128, 87])}
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

    # convert chart into data_temp
    # chart_temp is a simulation of chart, the 4 columns represent the 4 Keys in game
    # 0 for no notes
    # 1 for note
    # 2 for long note start
    # 3 for long note end
    data_temp["label"] = torch.zeros([beat_num, 256])
    chart_temp = torch.zeros(beat_num, 4)
    for note in chart["note"]:
        if "sound" not in note:
            if "endbeat" in note:
                chart_temp[note2beatindex(note["beat"]), note["column"]] = 2
                chart_temp[note2beatindex(note["endbeat"]), note["column"]] = 3
            else:
                chart_temp[note2beatindex(note["beat"]), note["column"]] = 1
    for i in range(beat_num):
        temp_index = chart_temp[i][0] * 1 + chart_temp[i][1] * 4 + chart_temp[i][2] * 16 + chart_temp[i][3] * 64
        data_temp["label"][i][int(temp_index)] = 1

    # embed data_temp into data
    data[index] = data_temp
    counter = counter + 1

print(data[0])

# save data as data.json
output = open("data.pkl", "wb")
pickle.dump(data, output)
output.close()
print(f"already save data into data.pkl, there are {counter} data")
