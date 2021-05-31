import numpy as np
data_path = 'C:/Users/joper/PycharmProjects/dimRating/data/'  # set path to data folder
# load data
y = np.loadtxt(data_path + 'spose_embedding_49d_sorted.txt')  # load y; path to folder resources
    # load indices of previously rated 20 images
test_ref_imgs = list(np.arange(20)*3)
for ind, img_code in enumerate(test_ref_imgs):  # format to four digits with leading 0s (like on website)
    test_ref_imgs[ind] = str(img_code).zfill(4)
    # load/generate list of 48 not-ref spose image id's
test_non_ref_imgs = list(np.arange(20)*5)
for ind, img_code in enumerate(test_non_ref_imgs):  # format to four digits with leading 0s (like on website)
    test_non_ref_imgs[ind] = str(img_code).zfill(4)
    # load list of 48 new image id's
test_new_imgs = list(np.arange(20) * 7)
for ind, img_code in enumerate(test_new_imgs):  # format to four digits with leading 0s (like on website)
    test_new_imgs[ind] = str(img_code).zfill(4)
# initialize experimental trial matrix: mat = np.empty((116,2)), mat = np.nan

# loop over all dims
    # for dim_id in range(np.size(y,1)):
# combine 48 and 48 to all_no-feedback_trials and randomize
# randomize 20 trials
# add 5 trials from no feedback
# add 1 trial from feedback
# add true_dim_score for feedback img

# randomization
# for 5 trials: no feedback (96 trials)
# for 6th trial: feedback
# blockwise shuffling trials 2-6? range(steps=6)
# in psychopy: if true_dim_score, then run feedback routine, else continue

# save as csv
