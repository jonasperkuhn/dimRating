import numpy as np
import pickle
import random
from shutil import copyfile, copytree

# set path to input data folder
path_input = 'C:/Users/joper/PycharmProjects/dimRating/data/'
# set path to offline experiment, where screenshots are generated for online use
path_exp_screenshot = 'C:/Private/Studium/Studium Leipzig/Masterarbeit/DimRating/Psychopy Screenshot/resources/'
# set path to final online experiments folder (one exp per dim)
path_exps_final = 'C:/Private/Studium/Studium Leipzig/Masterarbeit/DimRating/final exps/'
# set path to dimRating output folder, where percentiles will be saved
path_ptiles = 'C:/Users/joper/PycharmProjects/dimRating/output/percentiles/'

# load/create data
spose, stim_imgs_20, stim_imgs_48_nonref, stim_imgs_48_new = get_data()

# set params
n_trials_train = 190  # number of total training trials
n_trials_train_nofb = 10  # number of trials were no feedback will be given anymore
n_anchors = 7  # number of dim scale anchors
n_anchors_pos = n_anchors - 1
anchor_step_scaled = 1/(n_anchors_pos)
n_anchor_imgs = 15  # max number of imgs per anchor
n_anchor_imgs_insp_highest = 36  # max number of imgs displayed in the inspect_highest_anchor_imgs routine
zero_cutoff = 0.3  # scale cutoff, to calculate percentiles separately below and above cutoff for a percentile scale that also differentiates the high values (else biased by many low values)
n_blocks_train = 3  # number of training blocks
n_blocks_exp = 2  # number of experimental blocks: set to a number that 96, 20, and (96 + 20 =) 116 can be divided by -> 2 or 4
header = 'img_code,dim_score_true,feedback,dim_score_true_ptile'  # variable names in input csv's
random.seed(808)

# combine 48 and 48 to all_no-feedback_trials and randomize
trials_fb = stim_imgs_20  # don't random shuffle to keep alignment to indices
trials_nofb = stim_imgs_48_nonref + stim_imgs_48_new
random.shuffle(trials_nofb)

# calc trial numbers
n_trials = len(trials_fb + trials_nofb)
n_trials_per_block = int(n_trials / n_blocks_exp)
n_trials_fb_per_block = int(len(trials_fb) / n_blocks_exp)
n_trials_nofb_per_block = int(len(trials_nofb) / n_blocks_exp)

# loop over dims
for dim_id in range(np.size(spose, 1)):
    path_output = get_dim_paths()  # get output path
    dim_scores = spose[:, dim_id]  # select dimension specific data
    object_indices = list(np.arange(len(dim_scores)))  # create list of indices to convert to img codes

    # compute percentiles
    ptiles_all, ptiles_nonzero, img_ind_nonzero, img_ind_zero = get_ptiles()

    # generate anchor img links
    anchor_images_examples, n_anchor_imgs_very, anchor_dev = get_anchor_imgs()

    ### generate training trial csv
    # select training images set to sample from, without anchor images and 20 test images
    train_img_pop = [x for x in object_indices if (x not in anchor_images_examples) and (x not in stim_imgs_20)]
    # randomly select n_training_imgs-n_anchors images
    train_img_codes_sample = random.sample(train_img_pop, n_trials_train - n_anchors_pos)
    # split off training no-feedback block of (n_trials_train_nofb) trials, set feedback to 0, and save as csv
    trial_mat_nofb = np.zeros((n_trials_train_nofb, 4))
    train_img_codes_nofb = random.sample(train_img_codes_sample, n_trials_train_nofb)
    trial_mat_nofb[:, 0] = train_img_codes_nofb
    train_img_codes_nofb_ind = [int(x) for x in train_img_codes_nofb]  # convert img codes back to indices
    trial_mat_nofb[:, 1] = [dim_scores[i] for i in train_img_codes_nofb_ind]  # add true_dim_score
    trial_mat_nofb[:, 2] = 0  # set all feedback to 0
    trial_mat_nofb[:, 3] = [ptiles_all[i] for i in train_img_codes_nofb_ind]  # add true_dim_score_ptile
    fname = path_output + 'condition files/traintrials_nofb.csv'  # set file name
    np.savetxt(fname, trial_mat_nofb, delimiter=",", header=header, comments='')   # save as .csv
    # add anchor training images to list, and shuffle
    train_img_codes_sample2 = [x for x in train_img_codes_sample if (x not in train_img_codes_nofb)]
    train_img_codes_fb = train_img_codes_sample2 + anchor_images_examples
    random.shuffle(train_img_codes_fb)
    # create trial matrix and fill with img_codes, corresponding dim_scores, and feedback=1
    trial_mat_fb = np.zeros((n_trials_train - n_trials_train_nofb, 4))
    trial_mat_fb[:, 0] = train_img_codes_fb
    train_img_codes_fb_ind = [int(x) for x in train_img_codes_fb]  # convert img codes back to indices
    trial_mat_fb[:, 1] = [dim_scores[i] for i in train_img_codes_fb_ind]  # add true_dim_score for feedback img
    trial_mat_fb[:, 2] = 1  # set all feedback to 1
    trial_mat_fb[:, 3] = [ptiles_all[i] for i in train_img_codes_fb_ind]  # add true_dim_score_ptile
    # split feedback training trials in blocks and save trial files
    trial_mat_list = np.split(trial_mat_fb, n_blocks_train)
    for block, trial_mat_split in enumerate(trial_mat_list):
        fname = path_output + 'condition files/traintrials_fb_block' + str(block) + '.csv'  # set file name
        np.savetxt(fname, trial_mat_split, delimiter=",", header=header, comments='')   # save as .csv
        # randomization per participant in psychopy, not here
    # get list of all training img codes
    stim_imgs_train = train_img_codes_fb + train_img_codes_nofb

    ### generate exp trial csv
    range_scores_nonzero = max(dim_scores) - min(dim_scores[np.where(dim_scores > 0.3)])
    # create trial matrix blockwise
    for block in range(n_blocks_exp):
        # initialize experimental trial matrix
        trial_mat_exp_block = np.zeros((n_trials_per_block, 4))
        # add 20/n_blocks trials
        trials_fb_block = range(block * n_trials_fb_per_block, (block + 1) * n_trials_fb_per_block)
        trial_mat_exp_block[0:n_trials_fb_per_block, 0] = [trials_fb[i] for i in trials_fb_block]
        # add true_dim_score for feedback img
        trial_mat_exp_block[0:n_trials_fb_per_block, 1] = [dim_scores[int(stim_imgs_20[i])] for i in trials_fb_block]
        # set feedback to 1 for feedback img
        trial_mat_exp_block[0:n_trials_fb_per_block, 2] = 1
        # add true_dim_score_ptile for feedback img
        trial_mat_exp_block[0:n_trials_fb_per_block, 3] = [ptiles_all[int(stim_imgs_20[i])] for i in trials_fb_block]
        # add 96/n_blocks feedback trials
        trials_nofb_block = range(block * n_trials_nofb_per_block, (block + 1) * n_trials_nofb_per_block)
        trial_mat_exp_block[n_trials_fb_per_block:n_trials_per_block, 0] = [trials_nofb[i] for i in trials_nofb_block]
        # save as csv
        fname = path_output + 'condition files/exptrials_block' + str(block) + '.csv'
        np.savetxt(fname, trial_mat_exp_block, delimiter=",", header=header, comments='')

    ### get all possible imgs for last (=highest) anchor
    # select after creating training trials, to avoid excluding all high imgs from training
    img_codes_inspect_highest = [img_ind_nonzero[img_code] for img_code in np.argsort(anchor_dev)
                                 if img_code not in stim_imgs_20
                                 and img_code not in stim_imgs_train
                                 ][0: min(n_anchor_imgs_insp_highest, n_anchor_imgs_very)]
    # save as pickle
    with open(path_exp_screenshot + 'anchor_img_codes/img_codes_insp_highest_' + str(dim_id) + '.pkl', 'wb') as f:
        pickle.dump(img_codes_inspect_highest, f)

    ### select stimulus images and copy them to exp folder
    # first for training and test images
    trial_img_list = stim_imgs_train + stim_imgs_20 + stim_imgs_48_nonref + stim_imgs_48_new
    # copy selected trial images to exp resources folder
    for img_code in trial_img_list:
        copyfile(path_input + 'test images/' + str(img_code) + '.jpg', path_output + 'test images/' + str(img_code) + '.jpg')
    print(dim_id)
