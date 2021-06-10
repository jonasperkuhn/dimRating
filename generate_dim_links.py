import numpy as np
# set params
path_data = 'C:/Private/Studium/Studium Leipzig/Masterarbeit/DimRating/Psychopy/resources/'  # set path to resources folder
dim_id = 48  # set dim id
n_anchors = 6  # count of anchors
n_anchor_imgs = 12
zero_cutoff = 0.3
# load data
spose = np.loadtxt(path_data + 'spose_embedding_49d_sorted.txt')  # load y; path to folder resources
stim_imgs_20 = np.loadtxt(path_data + 'ref_imgs_20.txt')  # load test stim images (20 ref imgs)
stim_imgs_20 = [int(i) for i in list(stim_imgs_20)]  # convert to list of integers
# load training images and convert to list of integers
stim_imgs_train0 = np.loadtxt(path_data + 'condition_files/traintrials_fb_block0.csv', delimiter=',', skiprows=1)[:,0]
stim_imgs_train1 = np.loadtxt(path_data + 'condition_files/traintrials_fb_block1.csv', delimiter=',', skiprows=1)[:,0]
stim_imgs_train = np.concatenate((stim_imgs_train0,stim_imgs_train1))  # combine data from both blocks
stim_imgs_train = [int(i) for i in list(stim_imgs_train)]

# select scores of relevant dimension
dim_scores = spose[:, dim_id]
# initialize img_code dict
img_codes = dict()

# get set of not-at-all images
img_ind_zero = list(np.where(dim_scores <= 0.1)[0])  # select imgs below 0.1
anchor_imgs_notatall = np.random.choice(img_ind_zero, n_anchor_imgs, replace=False)  # randomly choose 10 imgs as anchor imgs

# get non-zero images
img_ind_nonzero = list(np.where(dim_scores > zero_cutoff)[0])  # get indices
dim_scores_nonzero = [dim_scores[ind] for ind in img_ind_nonzero]  # get dim scores corresponding to nonzero imgs
n_anchor_imgs_very = int(len(img_ind_nonzero) / n_anchors / 2)  # get maximum nr of images for highest anchor
ptiles = [(len(list(np.where(np.array(dim_scores_nonzero) <= score)[0])) /
           len(dim_scores_nonzero)) * 100 for score in dim_scores_nonzero]  # convert scores to percentiles
# extract image codes for each anchor range, and sort from highest to lowest
for i_anchor in range(n_anchors):
    # determine anchor percentile
    ptile_anchor = i_anchor / (n_anchors-1) * 100
    # calculate percentile deviance of each percentile
    anchor_dev = [np.abs(ptile - ptile_anchor) for ptile in ptiles]
    # select n_anchor_imgs of lowest deviating percentiles
    img_codes_closest = [img_ind_nonzero[img] for img in np.argsort(anchor_dev)][0: n_anchor_imgs]
    # for very high and very low anchor, choose max. n_anchor_imgs_very imgs, to avoid img overlap to other anchors
    if i_anchor in [0, n_anchors-1]:
        img_codes_closest = [img_ind_nonzero[img]
                             for img in np.argsort(anchor_dev)][0:min(n_anchor_imgs, n_anchor_imgs_very)]
    # remove training images and previously rated 20 images (because they will be tested)
    img_codes[i_anchor] = [img_code for img_code in img_codes_closest
                           if img_code not in stim_imgs_20
                           and img_code not in stim_imgs_train]
# get all possible imgs for last (=highest) anchor
img_codes_inspect_highest = [img_ind_nonzero[img] for img in np.argsort(anchor_dev)][0: n_anchor_imgs_very]
