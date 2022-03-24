## Procedure for constructing separate dimension rating experiments
1.	Get folders of dimRating (Psychopy Labeled) and dimRating_screenshot (Psychopy Screenshot)
2.	Generate template experiment folder 'dimrating_template_folder'
- Add template experiment (dimRating.psyexp)
- Add folder ‘resources’ to template folder with:
  - folder ‘condition files’ with 2 condition files:
    - `choose_train_blocks.csv`
    - `choose_exp_blocks.csv`
  - folder ‘screenshots’ (empty)
  - folder ‘test images’ (empty)
  - mouse-cursor-img.png
3.	Create separate experiments for each dimension on prolific
- Add completion link, which redirects to prolific, to template dimRating.psyexp
4.	Run `generate_dimrating_input_data.py`
- Generates trials, anchor images, percentiles
- Copies separate dim experiment folders from template folder
- Copies relevant files (incl. trial csv’s and stimuli) into experiment folders
5.	Take screenshots of anchor images and trial, offline via dimRating_screenshot.psyexp
- 4 screenshots:
  - Screenshot_highest.png (of all very high images)
  - Screenshot_insp.png (of whole dimension of anchor images shown during instructions)
  - Screenshot_anchor_imgs.png (of short version of dimension shown during trial)
  - Screenshot_trial.png (of trial, to show during instructions)
- For each dim separately
- Then manually copy screenshots into previously generated exp folders
6.	Manually change dim_label (and optionally dim_id) at first slide of dimRating.psyexp for each dimension
7.	Commit each experiment folder to pavlovia/gitlab
a.	Set exps to public
8.	On pavlovia: set exps to ‘running’ and ‘save incomplete data’ and ’store data in database’
9.	Copy url’s to separate prolific exps
10.	Test on Prolific via preview, then run

