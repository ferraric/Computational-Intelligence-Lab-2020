# Sentiment Classification - Computational Intelligence Lab 2020

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![Build Status](https://travis-ci.com/ferraric/Computational-Intelligence-Lab-2020.svg?token=T9puYMxv2xj4sUZv4Vzc&branch=master)](https://travis-ci.com/ferraric/Computational-Intelligence-Lab-2020)

## Description   
This is a project that was done in the Computational Intelligence Lab 2020 at ETH Zurich (see [Course website](http://www.da.inf.ethz.ch/teaching/2020/CIL/)).
Specifically we are doing a sentiment analysis of twitter data and classify them into positive and negative sentiments (see [Kaggle competition](https://www.kaggle.com/c/cil-text-classification-2020)). 

## Setup 
Download data from: https://polybox.ethz.ch/index.php/s/MxU3xzbLLKytwRT

```
# clone project   
git clone https://github.com/ferraric/Computational-Intelligence-Lab-2020   

# install project   
cd Computational-Intelligence-Lab-2020    

# move data into the data folder
mkdir data
mv .../downloaded-data-folder /data

# --> if on Leonhard: install your python virtual environment into Computational-Intelligence-Lab-2020/venv
python3 -m venv /cluster/home/.../Computational-Intelligence-Lab-2020/venv
source ./init_leonhard.sh

# --> if local: 
pip install -r requirements.txt

 ```  
We use pre-commit hooks to format our code to comply with black and pep8. If you want to contribute execute: 
```pre-commit install```
 

## Testing
To test the model on a trained checkpoint, run your main with the corresponding config file and add the -t flag which is the path to the checkpoint. 


## Rule Approach 
To reproduce the experiments, train a BERT baseline model. Move the generated validation_data.txt and validation_labels.txt into a local folder, for example into data/rules. 

Per default, only the parenthesis rule is applied. *note to self: handle command line -r rules if want to add more rules*


First create the tweets which have patterns of the rules removed:

```rules/main.py -d "validation_data_path" -l "validation_labels_path" -s "save_path"```

For example: ```rules/main.py -d "data/rules/validation_data.txt" -l "data/rules/validation_labels.txt" -s "data/rules/tweets_parenthesis_rule.txt"```


To compare the performance of BERT:

Test BERT on the validation_data.txt. To do this, change the test_path in the config file to path to the validation_data.txt file. Download the predictions from comet.ml - access on request. The checkpoint to test on should be the BERT baseline model you trained on before. This are the predictions on the unmodified tweets. 

Test BERT on the newly saved tweets where patterns of the rules are present. Change the test_path in the config file to the tweet txt file you created. Download the predictions from comet.ml - access on request. The checkpoint to test on should be the BERT baseline model you trained on before. This are the predictions on the unmodified tweets. 


Then run the main file with the corresponding predictions from BERT to get the accuracy and the confusion matrix of bert and the rule based predictions: 

```rules/main.py -d "validation_data_path" -l "validation_labels_path" -b "bert_predictions_path"```

For example: ```rules/main.py -d "data/rules/validation_data.txt" -l "data/rules/validation_labels.txt" -b "data/rules/test_predictions.csv"```
