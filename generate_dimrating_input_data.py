import numpy as np
import pickle
import random
from shutil import copyfile, copytree

def get_data(path_input):
    spose = np.loadtxt(path_input + 'spose_embedding_49d_sorted.txt')  # load true dim scores
    stim_imgs_20 = np.loadtxt(path_input + 'ref_imgs_20.txt')  # image codes of reused images from old experiment
    stim_imgs_20 = [int(i) for i in list(stim_imgs_20)]  # convert to list of integers
    # generate list of 48 not-ref spose image id's: define img codes
    stim_imgs_48_nonref = list(np.arange(start=2000, stop=2048))
    # generate list of 48 not-things spose image id's: define img codes
    stim_imgs_48_new = list(np.arange(start=3000, stop=3048))
    return spose, stim_imgs_20, stim_imgs_48_nonref, stim_imgs_48_new

def get_dim_paths(dim_id, path_exps_final):
    # copy template experiment folder n_dims times, to create separate experiments for each spose dimension
        # first create template experiment folder manually, with resources subfolders for condition files, screenshots,
        # and test images, as well as the mouse-cursor-img.png
    dim_exp_folder_path = path_exps_final + 'dimrating_dim' + str(dim_id) + '/'
    copytree(path_exps_final + 'dimrating_template_folder', dim_exp_folder_path)
    # define output path as resources folder of newly created experiment folder
    path_output = dim_exp_folder_path + 'resources/'
    return path_output

def get_ptiles(dim_scores, zero_cutoff, n_anchors_pos, dim_id, path_ptiles):
    # select imgs above cutoff
    img_ind_nonzero = list(np.where(dim_scores > zero_cutoff)[0])
    # select one training image for each anchor (first image of anchor)
    dim_scores_nonzero = [dim_scores[ind] for ind in img_ind_nonzero]  # get dim scores corresponding to nonzero imgs
    # compute percentiles for nonzero imgs for feedback
    ptiles_nonzero = [(len(list(np.where(np.array(dim_scores_nonzero) <= score)[0])) /
                       len(dim_scores_nonzero)) * 100 for score in dim_scores_nonzero]  # convert scores to percentiles
    ptiles_nonzero_scaled = [((n_anchors_pos - 1) * 1/n_anchors_pos * ptile + 100 * 1/n_anchors_pos) for
                             ptile in ptiles_nonzero]  # rescale to anchor_step_scaled-100 scale for non 'not at all' items
    # compute percentiles for zero imgs for feedback
    img_ind_zero = list(np.where(dim_scores <= zero_cutoff)[0])  # select imgs below cutoff
    dim_scores_zero = [dim_scores[ind] for ind in img_ind_zero]  # get dim scores corresponding to notatall imgs
    ptiles_zero = [(len(list(np.where(np.array(dim_scores_zero) <= score)[0])) /
                    len(dim_scores_zero)) * 100 for score in dim_scores_zero]  # convert scores to percentiles
    ptiles_zero_scaled = [1/n_anchors_pos * ptile for ptile in
                          ptiles_zero]  # rescale to 0-anchor_step_scaled scale for 'not at all' items
    # save ptiles in same order as all dim scores
    ptiles_all = list(range(len(dim_scores)))
    for nonzero_ind, all_ind in enumerate(img_ind_nonzero):
        ptiles_all[all_ind] = ptiles_nonzero_scaled[nonzero_ind]
    for zero_ind, all_ind in enumerate(img_ind_zero):
        ptiles_all[all_ind] = ptiles_zero_scaled[zero_ind]
    fname = path_ptiles + 'ptiles_dim' + str(dim_id) + '.csv'  # set file name
    np.savetxt(fname, ptiles_all, delimiter=",", comments='')  # save as .csv
    return ptiles_all, ptiles_nonzero, img_ind_nonzero, img_ind_zero

def get_anchor_imgs(n_anchor_imgs, ptiles_nonzero, img_ind_nonzero, img_ind_zero, n_anchors_pos, stim_imgs_20, path_exp_screenshot):
    # initialize anchor img code dict
    anchor_img_codes = {}
    anchor_images_examples = []
    # get maximum nr of images for highest anchor
    n_anchor_imgs_very = int(len(img_ind_nonzero) / n_anchors_pos / 2)
    # randomly choose n_anchor_imgs imgs of zero imgs (below cutoff) as anchor imgs
    anchor_img_codes[0] = np.random.choice(img_ind_zero, n_anchor_imgs, replace=False)
    # extract image codes for each anchor range, and sort from closest to furthest away
    for i_anchor in range(n_anchors_pos):
        # determine anchor percentile
        ptile_anchor = i_anchor / (n_anchors_pos - 1) * 100
        # calculate percentile deviance of each percentile
        anchor_dev = [np.abs(ptile - ptile_anchor) for ptile in ptiles_nonzero]
        # select n_anchor_imgs of lowest deviating percentiles
        # for very high and very low anchor, choose max. n_anchor_imgs_very imgs, to avoid img overlap to other anchors
        if i_anchor in [0, n_anchors_pos - 1]:
            img_codes_closest = [img_ind_nonzero[img]
                                 for img in np.argsort(anchor_dev)][0:min(n_anchor_imgs, n_anchor_imgs_very)]
        # for other anchors, take all n_anchor_imgs imgs (no threat of overlap)
        else:
            img_codes_closest = [img_ind_nonzero[img] for img in np.argsort(anchor_dev)][0: n_anchor_imgs]
        # remove previously rated 20 images (because they will be tested)
        anchor_img_codes[i_anchor + 1] = [int(img_code) for img_code in img_codes_closest
                                          if img_code not in stim_imgs_20]

        anchor_images_examples.append(anchor_img_codes[i_anchor + 1][-1])  # append last (furthest away) anchor image
    # save codes of each anchor as pickle
    with open(path_exp_screenshot + 'anchor_img_codes/anchor_img_codes_' + str(dim_id) + '.pkl', 'wb') as f:
        pickle.dump(anchor_img_codes, f)
    return anchor_images_examples, n_anchor_imgs_very, anchor_dev

def create_trial_mat(fb: int, n_trials_train_mat, train_img_codes, ptiles_all, dim_scores):
    trial_mat = np.zeros((n_trials_train_mat, 4))
    trial_mat[:, 0] = train_img_codes
    train_img_codes_ind = [int(x) for x in train_img_codes]  # convert img codes back to indices
    trial_mat[:, 1] = [dim_scores[i] for i in train_img_codes_ind]  # add true_dim_score
    trial_mat[:, 2] = fb  # set all feedback to fb
    trial_mat[:, 3] = [ptiles_all[i] for i in train_img_codes_ind]  # add true_dim_score_ptile
    return trial_mat

def save_train_trials(object_indices, anchor_images_examples, stim_imgs_20, n_trials_train, n_trials_train_nofb,
                      n_anchors_pos, n_blocks_train, header, path_output):
    # select training images set to sample from, without anchor images and 20 test images
    train_img_pop = [x for x in object_indices if (x not in anchor_images_examples) and (x not in stim_imgs_20)]
    # randomly select n_training_imgs-n_anchors images
    train_img_codes_sample = random.sample(train_img_pop, n_trials_train - n_anchors_pos)
    # create no-feedback training trials and save as csv
    train_img_codes_nofb = random.sample(train_img_codes_sample, n_trials_train_nofb)
    trial_mat_nofb = create_trial_mat(fb = 0, n_trials_train_mat = n_trials_train_nofb,
        train_img_codes = train_img_codes_nofb, ptiles_all = ptiles_all, dim_scores = dim_scores)
    fname = path_output + 'condition files/traintrials_nofb.csv'  # set file name
    np.savetxt(fname, trial_mat_nofb, delimiter=",", header=header, comments='')   # save as .csv
    # add anchor training images to list, and shuffle
    train_img_codes_sample2 = [x for x in train_img_codes_sample if (x not in train_img_codes_nofb)]
    train_img_codes_fb = train_img_codes_sample2 + anchor_images_examples
    random.shuffle(train_img_codes_fb)
    # create trial matrix and fill with img_codes, corresponding dim_scores, and feedback=1
    trial_mat_fb = create_trial_mat(fb = 1, n_trials_train_mat = n_trials_train - n_trials_train_nofb,
        train_img_codes = train_img_codes_fb, ptiles_all = ptiles_all, dim_scores = dim_scores)
        # split feedback training trials in blocks and save trial files
    trial_mat_list = np.split(trial_mat_fb, n_blocks_train)
    for block, trial_mat_split in enumerate(trial_mat_list):
        fname = path_output + 'condition files/traintrials_fb_block' + str(block) + '.csv'  # set file name
        np.savetxt(fname, trial_mat_split, delimiter=",", header=header, comments='')   # save as .csv
        # randomization per participant in psychopy, not here
    # get list of all training img codes
    stim_imgs_train = train_img_codes_fb + train_img_codes_nofb
    return stim_imgs_train

def save_exp_trials(dim_scores, n_blocks_exp, n_trials_per_block, n_trials_fb_per_block, n_trials_nofb_per_block,
                    trials_fb, trials_nofb, ptiles_all, stim_imgs_20, header, path_output):
    range_scores_nonzero = max(dim_scores) - min(dim_scores[np.where(dim_scores > 0.3)])
    # create exp trial matrix blockwise
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
    return


# set path to input data folder
path_input = 'C:/Users/joper/PycharmProjects/dimRating/data/'
# set path to offline experiment, where screenshots are generated for online use
path_exp_screenshot = 'C:/Private/Studium/Studium Leipzig/Masterarbeit/DimRating/Psychopy Screenshot/resources/'
# set path to final online experiments folder (one exp per dim)
path_exps_final = 'C:/Private/Studium/Studium Leipzig/Masterarbeit/DimRating/final exps/'
# set path to dimRating output folder, where percentiles will be saved
path_ptiles = 'C:/Users/joper/PycharmProjects/dimRating/output/percentiles/'

# set params
n_trials_train = 190  # number of total training trials
n_trials_train_nofb = 10  # number of trials were no feedback will be given anymore
n_anchors = 7  # number of dim scale anchors
n_anchors_pos = n_anchors - 1
n_anchor_imgs = 15  # max number of imgs per anchor
n_anchor_imgs_insp_highest = 36  # max number of imgs displayed in the inspect_highest_anchor_imgs routine
zero_cutoff = 0.3  # scale cutoff, to calculate percentiles separately below and above cutoff for a percentile scale that also differentiates the high values (else biased by many low values)
n_blocks_train = 3  # number of training blocks
n_blocks_exp = 2  # number of experimental blocks: set to a number that 96, 20, and (96 + 20 =) 116 can be divided by -> 2 or 4
header = 'img_code,dim_score_true,feedback,dim_score_true_ptile'  # variable names in input csv's
random.seed(808)


# load/create data
spose, stim_imgs_20, stim_imgs_48_nonref, stim_imgs_48_new = get_data(path_input=path_input)
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
    path_output = get_dim_paths(dim_id=dim_id, path_exps_final=path_exps_final)  # get output path
    dim_scores = spose[:, dim_id]  # select dimension specific data
    object_indices = list(np.arange(len(dim_scores)))  # create list of indices to convert to img codes
    # compute percentiles
    ptiles_all, ptiles_nonzero, img_ind_nonzero, img_ind_zero = get_ptiles(dim_scores=dim_scores,
        zero_cutoff=zero_cutoff, n_anchors_pos=n_anchors_pos, dim_id=dim_id, path_ptiles=path_ptiles)
    # generate and save anchor img links
    anchor_images_examples, n_anchor_imgs_very, anchor_dev_highest = get_anchor_imgs(n_anchor_imgs=n_anchor_imgs,
        ptiles_nonzero=ptiles_nonzero, img_ind_nonzero=img_ind_nonzero, img_ind_zero=img_ind_zero,
            n_anchors_pos=n_anchors_pos, stim_imgs_20=stim_imgs_20, path_exp_screenshot=path_exp_screenshot)
    # generate and save training trial csv's
    stim_imgs_train = save_train_trials(object_indices=object_indices, anchor_images_examples=anchor_images_examples,
        stim_imgs_20=stim_imgs_20, n_trials_train=n_trials_train, n_trials_train_nofb=n_trials_train_nofb,
            n_anchors_pos=n_anchors_pos, n_blocks_train=n_blocks_train, header=header, path_output=path_output)
    # generate exp trial csv
    save_exp_trials(dim_scores=dim_scores, n_blocks_exp=n_blocks_exp, n_trials_per_block=n_trials_per_block,
        n_trials_fb_per_block=n_trials_fb_per_block, n_trials_nofb_per_block=n_trials_nofb_per_block, trials_fb=trials_fb,
            trials_nofb=trials_nofb, ptiles_all=ptiles_all, stim_imgs_20=stim_imgs_20, header=header, path_output=path_output)

    # get all possible imgs for last (=highest) anchor
    # select after creating training trials, to avoid excluding all high imgs from training
    img_codes_inspect_highest = [img_ind_nonzero[img_code] for img_code in np.argsort(anchor_dev_highest)
                                 if img_code not in stim_imgs_20
                                 and img_code not in stim_imgs_train
                                 ][0: min(n_anchor_imgs_insp_highest, n_anchor_imgs_very)]
    # save as pickle
    with open(path_exp_screenshot + 'anchor_img_codes/img_codes_insp_highest_' + str(dim_id) + '.pkl', 'wb') as f:
        pickle.dump(img_codes_inspect_highest, f)

    # select stimulus images and copy them to exp folder
    # combine training and test images
    trial_img_list = stim_imgs_train + stim_imgs_20 + stim_imgs_48_nonref + stim_imgs_48_new
    # copy selected trial images to exp resources folder
    for img_code in trial_img_list:
        copyfile(path_input + 'test images/' + str(img_code) + '.jpg', path_output + 'test images/' + str(img_code) + '.jpg')
    print(dim_id)
