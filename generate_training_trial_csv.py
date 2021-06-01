import numpy as np
import random
# set params
data_path = 'C:/Users/joper/PycharmProjects/dimRating/data/'  # set path to data folder
n_training_imgs = 120
n_anchors = 7
# load data
y = np.loadtxt(data_path + 'spose_embedding_49d_sorted.txt')  # load true dim scores; path to folder resources
test_ref_imgs = list(np.arange(20)*3)
for ind, img_code in enumerate(test_ref_imgs):  # format to four digits with leading 0s (like on website)
    test_ref_imgs[ind] = str(img_code).zfill(4)
# loop over dims
for dim_id in range(np.size(y,1)):
    # select data relevant for dimension and add img codes
    dim_scores = y[:, dim_id]
    object_indices = list(np.arange(len(dim_scores)))  # create list of indices to convert to img codes
    for ind, img_code in enumerate(object_indices):  # format to four digits with leading 0s (like on website)
        object_indices[ind] = str(img_code).zfill(4)
    # dim_scores_ind = np.vstack((dim_scores, np.array(object_indices))).T  # currently not needed
    # select one training images for each anchor (first image of anchor)
    range_scores = max(dim_scores) - min(dim_scores)
    anchor_images_examples = []
    for i_anchor in range(n_anchors):
        # determine lower-bound score of anchor (e.g., 0.578 corresponds to 20%)
        dim_score_lowest = min(dim_scores) + range_scores * (i_anchor/n_anchors)
        # determine upper-bound score of anchor (e.g., 0.765 corresponds to 40%)
        dim_score_highest = min(dim_scores) + range_scores * ((i_anchor + 1)/n_anchors)
        # select indices of images between lowest (excl.) and highest (incl.) scoring image in anchor range
        anchor_images = list(np.where((dim_scores > dim_score_lowest) & (dim_scores <= dim_score_highest))[0])
        # randomly select one image for each anchor and append to anchor images list, formatted as 4-digits number
        anchor_images_examples.append(str(random.choice(anchor_images)).zfill(4))
        # todo: check back with img selection in generate_dim_links.py
    # select training images set to sample from, without anchor images and 20 test images
    train_img_pop = [x for x in object_indices if (x not in anchor_images_examples) and (x not in test_ref_imgs)]
    # randomly select n_training_imgs-n_anchors images
    training_img_codes = random.sample(train_img_pop, n_training_imgs-n_anchors)
    # add anchor training images to list
    training_img_codes = training_img_codes + anchor_images_examples
    # todo: create csv with three cols: training_img_codes, true_dim_score, and feedback (in pandas)
    # set all feedback to 1
    # randomize per participant in psychopy, not here!
