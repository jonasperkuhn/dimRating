import numpy as np
# set params
data_path = 'C:/Users/joper/PycharmProjects/dimRating/data/'  # set path to data folder
n_dim = 49
n_anchors = 7  # anchors plus one to get bins
# load data
y = np.loadtxt(data_path + 'spose_embedding_49d_sorted.txt')  # load y
dim_imgs = dict()
for i_dim in range(n_dim):
    dim_scores = y[:, i_dim]
    range_scores = max(dim_scores) - min(dim_scores)
    dim_dict = dict()
    for i_anchor in range(n_anchors):
        dim_score_lowest = min(dim_scores) + range_scores * (i_anchor/n_anchors)
        dim_score_highest = min(dim_scores) + range_scores * ((i_anchor + 1)/n_anchors)
        dim_dict[i_anchor] = np.where((dim_scores > dim_score_lowest) & (dim_scores <= dim_score_highest))[0]
    dim_imgs[i_dim] = dim_dict
