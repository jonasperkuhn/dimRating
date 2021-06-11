## end feedback routine, if feedback variable in conditions file = 0
if feedback == 0:
    continueRoutine = False # end routine

## reset scale position
dim_rater_fb.pos = (pos_scale_x, pos_scale_y)

## calculate performance and feedback
scale_steps = n_anchors + 1  # absolute steps are 7 (-1 to 5)
scale_pos_steps = n_anchors - 1  # positive scale goes from 0 to 5
# get true dim scores
if true_dim_score <= zero_cutoff:  # if true dim is 'not at all' (below cutoff)
    dim_score_true_ptile = -1
    pos_dim_true = pos_notatall_y - 0.02
else:
    i_score_true = np.abs(np.array(dim_scores_nonzero) - true_dim_score).argmin()  # find index of true dim score img
    dim_score_true_ptile = ptiles[i_score_true]  # select corresponding percentile
    pos_dim_true = (pos_anchors_y[0] + size_scale_y * dim_score_true_ptile / 100)
# get rated dim scores
if dim_rater_trial.getRating() is None:  # if rated 'not at all'
    score_rated_ptile = -1
    pos_dim_rated = pos_notatall_y - 0.02
else:
    score_rated = dim_rater_trial.getRating()
    score_rated_ptile = score_rated / scale_pos_steps * 100  # scale rated score to percentile range
    i_score_rated = np.abs(np.array(ptiles) - score_rated_ptile).argmin()  # find index of closest
    score_rated_spose = dim_scores_nonzero[i_score_rated]  # select corresponding percentile
    pos_dim_rated = (pos_anchors_y[0] + size_scale_y * score_rated_ptile / 100)
# determine difference between rated and true dim score in percent
diff_percent_ptile = round(abs(score_rated_ptile - dim_score_true_ptile))
# set percent deviation as string for display
deviation_message.text = (str(diff_percent_ptile) + "% off")
# determine the positions to highlight on slider scale
pos_diff = abs(pos_dim_rated - pos_dim_true)

## determine color of deviation rectangle
if diff_percent_ptile < 1:
    deviation_color = 'green'
    deviation_message.text = 'Correct!\nWell done!'
elif diff_percent_ptile < 5:
    deviation_color = 'yellowgreen'
elif diff_percent_ptile < 30:
    deviation_color = 'orange'
else:
    deviation_color = 'red'
