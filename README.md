# Sentiment Classification - Computational Intelligence Lab 2020

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![Build Status](https://travis-ci.com/ferraric/Computational-Intelligence-Lab-2020.svg?token=T9puYMxv2xj4sUZv4Vzc&branch=master)](https://travis-ci.com/ferraric/Computational-Intelligence-Lab-2020)

## Description   
This is a project that was done in the Computational Intelligence Lab 2020 at ETH Zurich (see [Course website](http://www.da.inf.ethz.ch/teaching/2020/CIL/)).
Specifically we are doing a sentiment analysis of twitter data and classify them into positive and negative sentiments (see [Kaggle competition](https://www.kaggle.com/c/cil-text-classification-2020)). Our team name was FrontRowCrew.

## Setup 
Download data from: http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip

```
# clone project   
git clone https://github.com/ferraric/Computational-Intelligence-Lab-2020   

# install project   
cd Computational-Intelligence-Lab-2020    

# move data into the data folder
mkdir data
mv path-to-downloaded-folder/downloaded-data-folder data

# --> if on Leonhard: install your python virtual environment into Computational-Intelligence-Lab-2020/venv
python3 -m venv ~/Computational-Intelligence-Lab-2020/venv
source ./init_leonhard.sh

# --> if local: 
pip install -r requirements.txt

 ```  

## Reproduce Exerperiments
To test the model on a trained checkpoint, run your main with the corresponding config file and add the -t flag which is the path to the checkpoint. 

### Baselines

#### Google Natural Language API

#### GloVe

#### BERT

### BERTweet

### Additional Data

### Ensemble Learning

### [Section to be named]
To reproduce the experiments, train a BERT baseline model. 

First create the tweets which have patterns of the rules removed:

```rules/main.py -d "validation_data_path" -l "validation_labels_path" -s "save_path"```


To compare the performance of BERT:

Test BERT on the validation_data.txt. To do this, change the test_path in the config file to path to the validation_data.txt file. Download the predictions (predictions are stored on comet.ml, access can be given on request). The checkpoint to test on should be the BERT baseline model you trained on before. This are the predictions on the unmodified tweets. 

Test BERT on the newly saved tweets where patterns of the rules are present. To do this, change the test_path in the config file to the tweet txt file you created. Download the predictions (predictions are stored on comet.ml, access can be given on request). The checkpoint to test on should be the BERT baseline model you trained on before. This are the predictions on the unmodified tweets. 


Then run the main file with the corresponding predictions from BERT to get the accuracy and the confusion matrix of bert and the rule based predictions: 

```rules/main.py -d "validation_data_path" -l "validation_labels_path" -b "bert_predictions_path"```


