import numpy as np
import random

# set params
path = 'C:/Users/joper/PycharmProjects/dimRating/'  # set path to data folder
n_trials_train = 190  # number of total training trials
n_trials_train_nofb = 10  # number of trials were no feedback will be given anymore
n_anchors = 6  # number of dim scale anchors
n_anchor_imgs = 12  # max number of imgs per anchor
zero_cutoff = 0.3
n_blocks = 3
random.seed(808)
header = 'img_code,dim_score_true,feedback'

# load data
y = np.loadtxt(path + 'data/spose_embedding_49d_sorted.txt')  # load true dim scores; path to folder resources
stim_imgs_20 = list(np.loadtxt(path + 'data/ref_imgs_20.txt'))
# loop over dims
for dim_id in range(np.size(y,1)):
    # select data relevant for dimension and add img codes
    dim_scores = y[:, dim_id]
    object_indices = list(np.arange(len(dim_scores)))  # create list of indices to convert to img codes
    # dim_scores_ind = np.vstack((dim_scores, np.array(object_indices))).T  # currently not needed
    img_ind_nonzero = list(np.where(dim_scores > zero_cutoff)[0])
    # select one training image for each anchor (first image of anchor)
    anchor_images_examples = []
    # get nchor imgs
    dim_scores_nonzero = [dim_scores[ind] for ind in img_ind_nonzero]  # get dim scores corresponding to nonzero imgs
    n_anchor_imgs_very = int(len(img_ind_nonzero) / n_anchors / 2)  # get maximum nr of images for highest anchor
    ptiles = [(len(list(np.where(np.array(dim_scores_nonzero) <= score)[0])) /
               len(dim_scores_nonzero)) * 100 for score in dim_scores_nonzero]  # convert scores to percentiles
    # extract image codes for each anchor range, and sort from closest to to furthest from anchor
    for i_anchor in range(n_anchors):
        # determine anchor percentile
        ptile_anchor = i_anchor / (n_anchors - 1) * 100
        # calculate percentile deviance of each percentile
        anchor_dev = [np.abs(ptile - ptile_anchor) for ptile in ptiles]
        # select n_anchor_imgs of lowest deviating percentiles
        img_codes_closest = [img_ind_nonzero[img] for img in np.argsort(anchor_dev)][0: n_anchor_imgs]
        # for very high and very low anchor, choose max. n_anchor_imgs_very imgs, to avoid img overlap to other anchors
        if i_anchor in [0, n_anchors - 1]:
            img_codes_closest = [img_ind_nonzero[img]
                                 for img in np.argsort(anchor_dev)][0:min(n_anchor_imgs, n_anchor_imgs_very)]
        # remove training images and previously rated 20 images (because they will be tested)
        anchor_images = [img_code for img_code in img_codes_closest
                               if img_code not in stim_imgs_20]
        anchor_images_examples.append(anchor_images[-1])  # append last (furthest away) anchor image
    # select training images set to sample from, without anchor images and 20 test images
    train_img_pop = [x for x in object_indices if (x not in anchor_images_examples) and (x not in stim_imgs_20)]
    # randomly select n_training_imgs-n_anchors images
    train_img_codes_sample = random.sample(train_img_pop, n_trials_train - n_anchors)
    # split off training no-feedback block of (n_trials_train_nofb) trials, set feedback to 0, and save as csv
    trial_mat_nofb = np.zeros((n_trials_train_nofb, 3))
    train_img_codes_nofb = random.sample(train_img_codes_sample, n_trials_train_nofb)
    trial_mat_nofb[:, 0] = train_img_codes_nofb
    train_img_codes_nofb_ind = [int(x) for x in train_img_codes_nofb]  # convert img codes back to indices
    trial_mat_nofb[:, 1] = [y[i, dim_id] for i in train_img_codes_nofb_ind]  # add true_dim_score
    fname = path + 'trial_csvs/dim' + str(dim_id) + '_traintrials_nofb.csv'  # set file name
    np.savetxt(fname, trial_mat_nofb, delimiter=",", header=header, comments='')   # save as .csv

    # add anchor training images to list, and shuffle
    train_img_codes_sample2 = [x for x in train_img_codes_sample if (x not in train_img_codes_nofb)]
    train_img_codes = train_img_codes_sample2 + anchor_images_examples
    random.shuffle(train_img_codes)
    # create trial matrix and fill with img_codes, corresponding dim_scores, and feedback=1
    trial_mat = np.zeros((n_trials_train - n_trials_train_nofb, 3))
    trial_mat[:, 0] = train_img_codes
    train_img_codes_ind = [int(x) for x in train_img_codes]  # convert img codes back to indices
    trial_mat[:, 1] = [y[i, dim_id] for i in train_img_codes_ind]  # add true_dim_score for feedback img
    trial_mat[:, 2] = 1  # set all feedback to 1
    # split feedback training trials in blocks and save trial files
    trial_mat_list = np.split(trial_mat, n_blocks)
    for block, trial_mat_split in enumerate(trial_mat_list):
        fname = path + 'trial_csvs/dim' + str(dim_id) + '_traintrials_fb_block' + str(block) + '.csv'  # set file name
        np.savetxt(fname, trial_mat_split, delimiter=",", header=header, comments='')   # save as .csv
    # randomization per participant in psychopy, not here!
    # in psychopy: compute true_dim_score_percent
    # in psychopy: format img codes according to website code (4 digits, starting from 0001)
