### feedback code
## end feedback routine, if feedback variable in conditions file = 0
if feedback == 0:
    continueRoutine = False # end routine
## reset scale position
dim_rater_fb.pos = (pos_scale_x, pos_scale_y)

n_anchors = 7
## calculate performance and feedback
# get rated dim scores and convert to percentile scale
score_rated = dim_rater_trial.getRating()
score_rated_ptile = (score_rated-1) / (n_anchors-1) * 100  # scale rated score to percentile range

# get scale position of rated and true dim scores
pos_dim_rated = (pos_anchors_y[0] + size_scale_y * score_rated_ptile / 100)
pos_dim_true = (pos_anchors_y[0] + size_scale_y * dim_score_true_ptile / 100)

# determine difference between rated and true dim score in percent
diff_percent_ptile = round(abs(score_rated_ptile - dim_score_true_ptile))
# collect deviations for all trials, to calc block performance
deviation_list.append(diff_percent_ptile)

# determine color of deviation rectangle
if diff_percent_ptile < 1:
    deviation_color = 'green'
    deviation_message.text = 'Correct!\nWell done!'
elif diff_percent_ptile < 5:
    deviation_color = 'yellowgreen'
elif diff_percent_ptile < 30:
    deviation_color = 'orange'
else:
    deviation_color = 'red'

# set percent deviation as string for display
deviation_message.text = (str(diff_percent_ptile) + "% off")
# determine the positions to highlight on slider scale
pos_diff = abs(pos_dim_rated - pos_dim_true)
# only for debugging:
print_scores_txt = (str(dim_score_true) + 'and' + str(dim_id) + 'and' + str(score_rated_ptile))
