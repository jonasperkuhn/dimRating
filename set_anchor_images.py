# code block in trial routine
## at begin experiment:
anchor_imgs = {}
for i_anchor in n_anchors:
    anchor_imgs_i = {}
    for i_img in n_anchor_images:
        anchor_imgs_i[i_img] = visual.ImageStim(
            win=win,
            name='dim_img_6_01',
            image='C:\\\\Private\\\\Studium\\\\Studium Leipzig\\\\Masterarbeit\\\\Präsi für Martin\\\\dimreg prediction r^2.png',
            mask=None,
            ori=0.0, pos=(pos_anchors_x[i_img], pos_anchors_y[i_anchor]), size=anchor_img_size,
            color=[1, 1, 1], colorSpace='rgb', opacity=None,
            flipHoriz=False, flipVert=False,
            texRes=128.0, interpolate=True, depth=-11.0)

## at begin routine:
# set after line defining trial components
# and before for loop that keeps track of finished components
for i_anchor in n_anchors:
    for i_img in n_anchor_images:
        trialComponents.append(anchor_imgs[i_anchor][i_img])

while continueRoutine:
    # for every image in xx:
    if dim_img_6_01.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
    # keep track of start time/frame for later
    dim_img_6_01.frameNStart = frameN  # exact frame index
    dim_img_6_01.tStart = t  # local t and not account for scr refresh
    dim_img_6_01.tStartRefresh = tThisFlipGlobal  # on global time
    win.timeOnFlip(dim_img_6_01, 'tStartRefresh')  # time at next scr refresh
    dim_img_6_01.setAutoDraw(True)