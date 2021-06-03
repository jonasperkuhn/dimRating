import numpy as np
# set params
data_path = 'C:/Users/joper/PycharmProjects/dimRating/data/'  # set path to resources folder
dim_id = 0  # set dim id
n_anchors = 6  # count of anchors
zero_cutoff = 0.3
# load data
y = np.loadtxt(data_path + 'spose_embedding_49d_sorted.txt')  # load y; path to folder resources
stim_imgs =
# select scores of relevant dimension
dim_scores = y[:, dim_id]
# initialize img_code dict
img_codes = dict()
# enter image codes of images below cut-off to 'notatall'
img_ind_zero = list(np.where(dim_scores <= zero_cutoff)[0])
# format to four digits, starting from 0001, with leading 0s (like on website), and save in dict
img_codes['notatall'] = [str(img_code+1).zfill(4) for img_code in img_ind_zero]
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
    # todo: remove previously rated 20 images (will be tested)
    # format to four digits, starting from 0001, with leading 0s (like on website), and save in dict
    img_codes[i_anchor] = [str(img_code+1).zfill(4) for img_code in img_codes_sorted]
