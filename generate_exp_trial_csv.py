data_path = 'C:/Users/joper/PycharmProjects/dimRating/data/'  # set path to data folder
# load data
dim_scores = np.loadtxt(data_path + 'spose_embedding_49d_sorted.txt')  # load y; path to folder resources
# load indices of previously rated 20 images
# load/generate list of 48 not-ref spose image id's
# load list of 48 new image id's
# combine to no-feedback trials
# generation of experimental trial matrix

# randomization
# for 5 trials: no feedback (96 trials)
# for 6th trial: feedback
# blockwise shuffling trials 2-6?
# in psychopy: if true_dim_score, then run feedback routine, else continue

# save as csv
