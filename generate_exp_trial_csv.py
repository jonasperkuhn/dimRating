import random
import numpy as np

# set params
path = 'C:/Users/joper/PycharmProjects/dimRating/'  # set path to data folder
zero_cutoff = 0.3
n_blocks = 2  # set to a number that 96, 20, and 116 can be divided by -> 2 or 4
random.seed(808)
header = 'img_code,true_dim_score,feedback'

# load data
y = np.loadtxt(path + 'data/spose_embedding_49d_sorted.txt')  # load y; path to folder resources
    # load indices of previously rated 20 images
test_ref_imgs = list(np.loadtxt(path + 'data/ref_imgs_20.txt'))
    # load/generate list of 48 not-ref spose image id's
test_non_ref_imgs = list(np.arange(48)*5)
    # load list of 48 new image id's
test_new_imgs = list(np.arange(48) * 7)
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
    # select dim data and compute range of values larger than zero-cutoff
    dim_scores = y[:, dim_id]
    range_scores_nonzero = max(dim_scores) - min(dim_scores[np.where(dim_scores > 0.3)])
    # create trial matrix blockwise
    for block in range(n_blocks):
        # initialize experimental trial matrix
        trial_mat = np.zeros((n_trials_per_block, 3))
        # add 20/n_blocks trials
        trials_fb_block = range(block*n_trials_fb_per_block, (block+1)*n_trials_fb_per_block)
        trial_mat[0:n_trials_fb_per_block, 0] = [trials_fb[i] for i in trials_fb_block]
        # add true_dim_score for feedback img
        trial_mat[0:n_trials_fb_per_block, 1] = [y[int(test_ref_imgs[i]), dim_id] for i in trials_fb_block]
        # set feedback to 1 for feedback img
        trial_mat[0:n_trials_fb_per_block, 2] = 1
        # add 96/n_blocks feedback trials
        trials_nofb_block = range(block*n_trials_nofb_per_block, (block+1)*n_trials_nofb_per_block)
        trial_mat[n_trials_fb_per_block:n_trials_per_block, 0] = [trials_nofb[i] for i in trials_nofb_block]
        # save as csv
        fname = path + 'trial_csvs/dim' + str(dim_id) + '_exptrials_block' + str(block) + '.csv'
        np.savetxt(fname, trial_mat, delimiter=",", header=header, comments='')
        # in psychopy: if true_dim_score, then run feedback routine, else continue
        # in psychopy: compute true_dim_score_percent
        # in psychopy: format img codes according to website code (4 digits, starting from 0001)
