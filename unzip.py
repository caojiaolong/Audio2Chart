'''
this file is used to unzip the .mcz files into ./data_unziped and rename them
'''
import gzip
import os
import zipfiles

data_raw = os.listdir("data_raw")
counter = 1
for mcz_file in data_raw:
    zip = zipfile.ZipFile("./data_raw/"+mcz_file)
    zip.extract()
