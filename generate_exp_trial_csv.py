import random
import numpy as np

# set params
path = 'C:/Users/joper/PycharmProjects/dimRating/'  # set path to data folder
n_blocks = 2  # set to a number that 96, 20, and 116 can be divided by -> 2 or 4
random.seed(808)
header = 'img_code,true_dim_score,feedback,true_dim_score_percent'

# load data
y = np.loadtxt(path + 'data/spose_embedding_49d_sorted.txt')  # load y; path to folder resources
    # load indices of previously rated 20 images
test_ref_imgs_ind = list(np.arange(20)*3)

test_ref_imgs = []
for ind, img_code in enumerate(test_ref_imgs_ind):  # format to four digits with leading 0s (like on website)
    test_ref_imgs.append(str(img_code).zfill(4))
    # load/generate list of 48 not-ref spose image id's
test_non_ref_imgs = list(np.arange(48)*5)
for ind, img_code in enumerate(test_non_ref_imgs):  # format to four digits with leading 0s (like on website)
    test_non_ref_imgs[ind] = str(img_code).zfill(4)
    # load list of 48 new image id's
test_new_imgs = list(np.arange(48) * 7)
for ind, img_code in enumerate(test_new_imgs):  # format to four digits with leading 0s (like on website)
    test_new_imgs[ind] = str(img_code).zfill(4)
# combine 48 and 48 to all_no-feedback_trials and randomize
trials_fb = test_ref_imgs  # don't random shuffle to keep alignment to indices
trials_nofb = test_non_ref_imgs + test_new_imgs
random.shuffle(trials_nofb)
# calc trial numbers
n_trials = len(trials_fb + trials_nofb)
n_trials_per_block = int(n_trials/n_blocks)
n_trials_fb_per_block = int(len(trials_fb) / n_blocks)
n_trials_nofb_per_block = int(len(trials_nofb) / n_blocks)
# loop over dims (for getting true_dim_scores)
for dim_id in range(np.size(y,1)):
    # select dim data and compute range
    dim_scores = y[:, dim_id]
    range_scores = max(dim_scores) - min(dim_scores)
    # create trial matrix blockwise
    for block in range(n_blocks):
        # initialize experimental trial matrix
        trial_mat = np.zeros((n_trials_per_block, 4))
        # add 20/n_blocks trials
        trials_fb_block = range(block*n_trials_fb_per_block, (block+1)*n_trials_fb_per_block)
        trial_mat[0:n_trials_fb_per_block, 0] = [trials_fb[i] for i in trials_fb_block]
        # add true_dim_score for feedback img
        trial_mat[0:n_trials_fb_per_block, 1] = [y[test_ref_imgs_ind[i], dim_id] for i in trials_fb_block]
        # set feedback to 1 for feedback img
        trial_mat[0:n_trials_fb_per_block, 2] = 1
        # convert true_dim_score to percent and save in row 4
        trial_mat[0:n_trials_fb_per_block, 3] = [score / range_scores * 100 for score in trial_mat[0:n_trials_fb_per_block, 1]]
        # add 96/n_blocks trials
        trials_nofb_block = range(block*n_trials_nofb_per_block, (block+1)*n_trials_nofb_per_block)
        trial_mat[n_trials_fb_per_block:n_trials_per_block, 0] = [trials_nofb[i] for i in trials_nofb_block]
        # save as csv
        fname = path + 'trial_csvs/dim' + str(dim_id) + '_exptrials_block' + str(block) + '.csv'
        np.savetxt(fname, trial_mat, delimiter=",", header=header)
        # in psychopy: if true_dim_score, then run feedback routine, else continue
