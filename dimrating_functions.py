import numpy as np
import pickle
import random
from shutil import copyfile, copytree

def get_data(path_input = path_input):
    spose = np.loadtxt(path_input + 'spose_embedding_49d_sorted.txt')  # load true dim scores
    stim_imgs_20 = np.loadtxt(path_input + 'ref_imgs_20.txt')  # image codes of reused images from old experiment
    stim_imgs_20 = [int(i) for i in list(stim_imgs_20)]  # convert to list of integers
    # generate list of 48 not-ref spose image id's: define img codes
    stim_imgs_48_nonref = list(np.arange(start=2000, stop=2048))
    # generate list of 48 not-things spose image id's: define img codes
    stim_imgs_48_new = list(np.arange(start=3000, stop=3048))
    return spose, stim_imgs_20, stim_imgs_48_nonref, stim_imgs_48_new

def get_dim_paths(dim_id = dim_id, path_exps_final = path_exps_final):
    # copy template experiment folder n_dims times, to create separate experiments for each spose dimension
        # first create template experiment folder manually, with resources subfolders for condition files, screenshots,
        # and test images, as well as the mouse-cursor-img.png
    dim_exp_folder_path = path_exps_final + 'dimrating_dim' + str(dim_id) + '/'
    copytree(path_exps_final + 'dimrating_template_folder', dim_exp_folder_path)
    # define output path as resources folder of newly created experiment folder
    path_output = dim_exp_folder_path + 'resources/'
    return path_output

def get_ptiles(dim_scores = dim_scores, zero_cutoff = zero_cutoff, n_anchors_pos = n_anchors_pos, dim_id = dim_id, path_ptiles = path_ptiles):
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
    fname = path_ptiles + 'ptiles_dim' + dim_id + '.csv'  # set file name
    np.savetxt(fname, ptiles_all, delimiter=",", comments='')  # save as .csv
    return ptiles_all, ptiles_nonzero, img_ind_nonzero, img_ind_zero

def get_anchor_imgs(n_anchor_imgs = n_anchor_imgs, ptiles_nonzero = ptiles_nonzero, img_ind_nonzero = img_ind_nonzero, img_ind_zero = img_ind_zero, n_anchors_pos = n_anchors_pos, stim_imgs_20 = stim_imgs_20, path_exp_screenshot = path_exp_screenshot):
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


