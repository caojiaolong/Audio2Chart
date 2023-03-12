'''
this file is used to unzip the .mcz files into ./data_unziped and rename them
'''
import os
import zipfile

data_raw = os.listdir("data_raw")
counter = 0
for i,mcz_file in enumerate(data_raw):
    counter=counter+1
    mcz_file_path = os.path.join("data_raw", mcz_file)
    zip = zipfile.ZipFile(mcz_file_path)
    for member in zip.namelist():
        zip.extract(member, 'data_unziped')
    os.rename("data_unziped/0","data_unziped/"+str(counter))

