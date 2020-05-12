# Computational Intelligence Lab 2020

### Important info for backtranslation augmentation of your dataset

The backtranslation code is taken from [UDA Github](https://github.com/google-research/uda/tree/master/back_translate). Since the code is almost a year old, the code they wrote runs on older tensorflow versions with different dependencies. Thus, to run the backtranslation on your dataset on Leonhard, you need to run first_time_setup.sh once which should setup a virtual environment which contains the necessary packages and doesn't interfere with possibly existing tensorflow installations etc.

After the initial setup, you need to edit the run.sh file and specify the name of your text file containing your sentences which should be located in the back_translate folder.

Run run_bt_job.sh to then submit the job to a cluster.