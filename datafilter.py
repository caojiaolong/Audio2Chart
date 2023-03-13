import os
import shutil
import json

folder_path = 'data_unziped'
counter = 0
for dir_name in os.listdir(folder_path):
    dir_path = os.path.join(folder_path, dir_name)
    for file_name in os.listdir(dir_path):
        source_file_path = os.path.join(dir_path, file_name)
        if file_name.endswith('.mc'):
            flag1 = 0
            f = open(source_file_path, 'r', encoding='utf-8')
            content = f.read()
            if '4K' in content:
                flag1 = 1
                if 'Lv.19' in content or 'Lv.20' in content or 'Lv.21' in content:
                    flag1 = 2
            else:
                flag1 = -1
            if flag1 == 2:
                file = open(source_file_path, 'r', encoding="utf-8")
                data = json.load(file)
                oggname = data["note"][-1]["sound"]
                oggpath = os.path.join(dir_path, oggname)
                if os.path.isfile(oggpath):
                    path = './data/'
                    folder_name1 = str(counter)
                    folder_path1 = os.path.join(path, folder_name1)
                    os.makedirs(folder_path1)
                    shutil.copy(source_file_path, folder_path1)
                    shutil.copy(oggpath, folder_path1)
                    counter = counter + 1
                break
