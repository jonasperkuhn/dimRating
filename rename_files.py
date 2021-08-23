# rename files in folder
import os
import csv
import numpy as np
path = 'C:/Users/joper/PycharmProjects/dimRating/data/'

# rename files with unique_id index
# load unique_ids (img names in sequence as spose embedding)
with open(path + 'unique_id.csv', newline='') as f:
    reader = csv.reader(f)
    unique_ids = [i[0] + '.jpg' for i in list(reader)]
path_imgs = path + 'images_old_names'
files = os.listdir(path_imgs)
for file in files:
    index = unique_ids.index(file)
    os.rename(os.path.join(path_imgs, file), os.path.join(path_imgs, ''.join([str(index), '.jpg'])))

# rename new images files with arbitrary code
path_imgs = path + 'images_old_names'
files = os.listdir(path_imgs)
for index, file in enumerate(files):
    os.rename(os.path.join(path_imgs, file), os.path.join(path_imgs, ''.join(['{:02d}'.format(index+1), '.jpg'])))

# get indices of ref test images from unique ids
# load list of reference test stimuli
with open(path + 'test_imgs_200_names.txt', newline='') as f:
    reader = csv.reader(f)
    test_imgs_200_named = [i[0] + '.jpg' for i in list(reader)]
# convert list of ref test stimuli from names to unique_id_indices
ref_imgs_200 = [unique_ids.index(img) for img in test_imgs_200_named]
fname = "ref_imgs_200.txt"
np.savetxt(fname, ref_imgs_200, delimiter=",", comments='')
