import numpy as np
# set params
data_path = 'C:/Users/joper/PycharmProjects/dimRating/data/'  # set path to data folder
n_dim = 49
n_anchors = 7  # anchors plus one to get bins
# load data
y = np.loadtxt(data_path + 'spose_embedding_49d_sorted.txt')  # load y; path to folder resources
dim_imgs = dict()
for i_dim in range(n_dim):
    dim_scores = y[:, i_dim]
    range_scores = max(dim_scores) - min(dim_scores)
    dim_dict = dict()
    for i_anchor in range(n_anchors):
        # determine lower-bound score of anchor (e.g., 0.578 corresponds to 20%)
        dim_score_lowest = min(dim_scores) + range_scores * (i_anchor/n_anchors)
        # determine upper-bound score of anchor (e.g., 0.765 corresponds to 40%)
        dim_score_highest = min(dim_scores) + range_scores * ((i_anchor + 1)/n_anchors)
        # select indices of images between lowest (excl.) and highest (incl.) scoring image in anchor range
        dim_dict[i_anchor] = list(np.where((dim_scores > dim_score_lowest) & (dim_scores <= dim_score_highest))[0])
        for ind, img_code in enumerate(dim_dict[i_anchor]):  # format to four digits with leading 0s (like on website)
            dim_dict[i_anchor][ind] = str(img_code).zfill(4)
    dim_imgs[i_dim] = dim_dict
