import numpy as np
# set params
path_data = 'C:/Private/Studium/Studium Leipzig/Masterarbeit/DimRating/Psychopy/resources/'  # set path to resources folder
dim_id = 44  # set dim id
n_anchors = 6  # count of anchors
zero_cutoff = 0.3
# load data
y = np.loadtxt(path_data + 'spose_embedding_49d_sorted.txt')  # load y; path to folder resources
stim_imgs_20 = np.loadtxt(path_data + 'ref_imgs_20.txt')
# select scores of relevant dimension
dim_scores = y[:, dim_id]
# initialize img_code dict
img_codes = dict()

# get not-at-all images
# enter image codes of images below cut-off to 'notatall'
img_codes_unsorted_zero = list(np.where(dim_scores <= zero_cutoff)[0])
# sort from highest to lowest score
sorted_indices_zero = list([np.argsort(dim_scores[img_codes_unsorted_zero])][0])
# select 10 lowest scoring images for not-at-all anchor
anchor_imgs_notatall = [img_codes_unsorted_zero[img_code] for img_code in sorted_indices_zero][-11:-1]

# alternative: select imgs below 0.1
img_ind_zero = list(np.where(dim_scores <= (zero_cutoff-0.2))[0])
# randomly choose n image codes below 0.1 as anchor imgs
anchor_imgs_notatall = np.random.choice(img_ind_zero, 12, replace=False)

# get non-zero images
# get indices
img_ind_nonzero = list(np.where(dim_scores > zero_cutoff)[0])
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
    # remove previously rated 20 images (because they will be tested)
    img_codes[i_anchor] = [img_code for img_code in img_codes_sorted if img_code not in stim_imgs_20]
