### feedback code
## end feedback routine, if feedback variable in conditions file = 0
if feedback == 0:
    continueRoutine = False # end routine
## reset scale position
dim_rater_fb.pos = (pos_scale_x, pos_scale_y)

## calculate performance and feedback
# get rated dim scores and convert to percentile scale
score_rated_centered = dim_rater_trial.getRating() / (n_anchors - 1) - 0.5  # centered to invert in js because scale is inverted online
score_rated_ptile = (score_rated_centered + 0.5) * 100  # scale rated score to percentile range

# get scale position of rated and true dim scores
pos_dim_rated = (pos_anchors_y[0] + size_scale_y * score_rated_ptile / 100)
pos_dim_true = (pos_anchors_y[0] + size_scale_y * dim_score_true_ptile / 100)

# determine difference between rated and true dim score in percent
diff_percent_ptile = round(abs(score_rated_ptile - dim_score_true_ptile))
# collect deviations for all trials, to calc block performance
deviation_list = deviation_list + [diff_percent_ptile]

# determine color of deviation rectangle
# psychopy rgb values are rgb/127.5-1
if diff_percent_ptile < 1:
    deviation_color = [-1,0,-1]  # 'green'
    deviation_message.text = 'Correct!\nWell done :)'
elif diff_percent_ptile < 5:
    deviation_color = [0.21,0.61,-0.61]  #'yellowgreen'
elif diff_percent_ptile < 30:
    deviation_color = [1,0.3,-1]  #'orange'
else:
    deviation_color = [1,-1,-1]  #'red'

# set percent deviation as string for display
deviation_message.text = (str(diff_percent_ptile) + "% off")
# determine the positions to highlight on slider scale
pos_diff = abs(pos_dim_rated - pos_dim_true)
# determine y position of deviation rectangle
pos_rect_dev_y = min(pos_dim_rated, pos_dim_true) + pos_diff/2
# only for debugging:
print_scores_txt = (str(dim_score_true_ptile) + 'and' + str(score_rated_centered) + 'and' + str(score_rated_ptile))
