import numpy as np
# set params
path_data = 'C:/Private/Studium/Studium Leipzig/Masterarbeit/DimRating/Psychopy/resources/'  # set path to resources folder
dim_id = 44  # set dim id
n_anchors = 6  # count of anchors
zero_cutoff = 0.3
# load data
y = np.loadtxt(path_data + 'spose_embedding_49d_sorted.txt')  # load y; path to folder resources
stim_imgs_20 = np.loadtxt(path_data + 'ref_imgs_20.txt')  # load test stim images (20 ref imgs)
stim_imgs_20 = [int(i) for i in list(stim_imgs_20)]  # convert to list of integers
# load training images and convert to list of integers
stim_imgs_train0 = np.loadtxt(path_data + 'condition_files/traintrials_fb_block0.csv', delimiter=',', skiprows=1)[:,0]
stim_imgs_train1 = np.loadtxt(path_data + 'condition_files/traintrials_fb_block1.csv', delimiter=',', skiprows=1)[:,0]
stim_imgs_train = np.concatenate((stim_imgs_train0,stim_imgs_train1))  # combine data from both blocks
stim_imgs_train = [int(i) for i in list(stim_imgs_train)]

# select scores of relevant dimension
dim_scores = y[:, dim_id]
# initialize img_code dict
img_codes = dict()

# get set of not-at-all images
img_ind_zero = list(np.where(dim_scores <= 0.1)[0])  # select imgs below 0.1
anchor_imgs_notatall = np.random.choice(img_ind_zero, 12, replace=False)  # randomly choose n=12 imgs as anchor imgs

# get non-zero images
img_ind_nonzero = list(np.where(dim_scores > zero_cutoff)[0])  # get indices
dim_scores_nonzero = [dim_scores[ind] for ind in img_ind_nonzero]
ptiles = [(len(list(np.where(np.array(dim_scores_nonzero) <= score)[0])) /
           len(dim_scores_nonzero)) * 100 for score in dim_scores_nonzero]  # convert to percentiles
# extract image codes for each anchor range, and sort from highest to lowest
for i_anchor in range(n_anchors):
    # determine lower-bound score of anchor (e.g., 0.578 corresponds to 20%)
    score_lowest = np.percentile(dim_scores[img_ind_nonzero], i_anchor / n_anchors * 100)
    # determine upper-bound score of anchor (e.g., 0.765 corresponds to 40%)
    score_highest = np.percentile(dim_scores[img_ind_nonzero], (i_anchor + 1) / n_anchors * 100)
    # select indices of images between lowest (excl.) and highest (incl.) scoring image in anchor range
    img_codes_unsorted = list(np.where((dim_scores > score_lowest) & (dim_scores <= score_highest))[0])
    # get indices of anchor images, sorted from highest to lowest
    sorted_indices = list([np.argsort(dim_scores[img_codes_unsorted])][0])
    img_codes_sorted = [img_codes_unsorted[img_code] for img_code in sorted_indices]
    # remove training images and previously rated 20 images (because they will be tested)
    img_codes[i_anchor] = [img_code for img_code in img_codes_sorted if img_code not in stim_imgs_20 and img_code not in stim_imgs_train]
