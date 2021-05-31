data_path = 'C:/Users/joper/PycharmProjects/dimRating/data/'  # set path to data folder
# load data
dim_scores = np.loadtxt(data_path + 'spose_embedding_49d_sorted.txt')  # load y; path to folder resources
    # load indices of previously rated 20 images
    # load/generate list of 48 not-ref spose image id's
    # load list of 48 new image id's

# initialize experimental trial matrix: mat = np.empty((116,2)), mat = np.nan

# loop over all dims
    # for dim_id in range(np.size(dim_scores,1)):
# combine 48 and 48 to all_no-feedback_trials and randomize
# randomize 20 trials
# add 5 trials from no feedback
# add 1 trial from feedback
# add true_dim_score for feedback img

# randomization
# for 5 trials: no feedback (96 trials)
# for 6th trial: feedback
# blockwise shuffling trials 2-6? range(steps=6)
# in psychopy: if true_dim_score, then run feedback routine, else continue

# save as csv
