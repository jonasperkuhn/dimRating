# rename files in folder
import os
import csv

path = 'C:/Users/joper/PycharmProjects/dimRating/data/'
# load unique_ids (img names in sequence as spose embedding)
with open(path + 'unique_id.csv', newline='') as f:
    reader = csv.reader(f)
    unique_ids = [i[0] + '.jpg' for i in list(reader)]
path_imgs = path + 'test images'
files = os.listdir(path_imgs)
# rename files with unique_id index
for file in files:
    index = unique_ids.index(file)
    os.rename(os.path.join(path_imgs, file), os.path.join(path_imgs, ''.join([str(index), '.jpg'])))

# rename new images files
import os
path = 'C:/Private/Studium/Studium Leipzig/Masterarbeit/DimRating/Stimuli/48new_200'
files = os.listdir(path)
for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpg'])))