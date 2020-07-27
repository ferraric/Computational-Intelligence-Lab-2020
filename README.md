# Sentiment Classification - Computational Intelligence Lab 2020

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![Build Status](https://travis-ci.com/ferraric/Computational-Intelligence-Lab-2020.svg?token=T9puYMxv2xj4sUZv4Vzc&branch=master)](https://travis-ci.com/ferraric/Computational-Intelligence-Lab-2020)

## Description   
This is a project that was done in the Computational Intelligence Lab 2020 at ETH Zurich (see [Course website](http://www.da.inf.ethz.ch/teaching/2020/CIL/)).
Specifically we are doing sentiment analysis of tweets and classify them into positive and negative sentiments (see [Kaggle competition](https://www.kaggle.com/c/cil-text-classification-2020)). Our team name was `FrontRowCrew`.

## Setup 
The experiments were run and tested with Python version 3.7.1.

Download data from: http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip


clone project  
```
git clone https://github.com/ferraric/Computational-Intelligence-Lab-2020   
```

install project 
```
cd Computational-Intelligence-Lab-2020    
```

move data into the data folder
```
mkdir data
mv path-to-downloaded-folder/downloaded-data-folder data
```
before running make sure that the source directory is recognized by your PYTHONPATH, for example do:
```
export PYTHONPATH=/path_to_source_directory/Computational-Intelligence-Lab-2020:$PYTHONPATH
```
If on Leonhard: install your python virtual environment into Computational-Intelligence-Lab-2020/venv
```
python3 -m venv ~/Computational-Intelligence-Lab-2020/venv
source ./init_leonhard.sh
```

If local:
```
pip install -r requirements.txt
 ```  
 
 Note that the pytorch version we used (pre-compiled with a specific cuda version) is not available for macOS. If you want to run it locally on a mac, change the pytorch version in the requirement file to the following:
 
 ```
 torch==1.5.0
 ```
 

## Reproduce Exerperiments
To log the experiment results we used Comet (https://www.comet.ml/docs/), a tensorboard like logger. Unfortunately we cannot make access to our experiment logs public. However, if access to the logs is needed, contact jeremyalain.scheurer@gmail.com.

All experiments can be run with the config option "use_comet_experiments": false. In that case, the logs and saved predictions are found in the same directory where the model checkpoint is saved. That path is built by concatenating the config option "model_save_directory" to the config option "experiment_name" and to the timestamp of execution start time. 
Ex: ```experiments/bert-baseline/20-07-25_12-25-02/```

To calculate the accuracy of a prediction file, run the following command:
```python ensemble/calculate_accuracy.py -p path-to-predictions.csv -l path-to-labels.csv```

The following holds for all models except Google Natural Language API and GloVe:

The hyperparameters `epochs`, `max_tokens_per_tweet`, `validation_size`, `validation_split_random_seed`, `batch_size`, `n_data_loader_workers`, `learning_rate` were unchanged for all runs with a particular model. How other hyperparameters were varied is described in the following sections.

When running an experiment, at the end of training, the provided test data are automatically predicted with the best saved checkpoint. If one needs to predict a set of tweets from an existing checkpoint, one needs to point the config option "test_tweets_path" to the corresponding tweets and provide the model checkpoint via the argument -t. Ex:
```python mains/bert.py -c configs/bert.json -t path-to-model-checkpoint/model_checkpoint.ckpt```

### Baselines

#### Google Natural Language API

Note that one needs a Google cloud account and credits to use this service. Make sure you set the variable GOOGLE_APPLICATION_CREDENTIALS to point to the json file containing your account credentials.
Also as a disclaimer we want to note that the Google Natural Language API is a service that costs money. One usually has a certain amount of free calls to the API (which we used). But we want to remind you to first check out what your API call limitations are in order to prevent a big bill at the end of the month.
```
export GOOGLE_APPLICATION_CREDENTIALS=path-to-your-account-credentials.json
```
```
python baselines/google_nlp_api.py -c configs/google_nlp_api.json
```


#### GloVe

To run the experiments for GloVe, one needs to download the code from [](https://github.com/dalab/lecture_cil_public/tree/master/exercises/2019/ex6) for generating the vocabulary and training the word embeddings.

In the README of that page, the instructions are given to build the vocabulary and the co-occurrence matrix for training. **Note that modifications have to be made to `build_vocab.sh` and `cooc.py` to use the full dataset.**

After building the co-occurrence matrix, one can run the training of the word embeddings by running

`python glove_solution.py`

In that script, the number of epochs can be specified.

After this is done, both the generated `vocab.pkl` and the `embeddings.npz` files have to be moved to the data folder defined in the Setup section.

To train a classifier using the GloVe embeddings, one has to run:

`python baselines/main_glove.py -c configs/glove_embeddings_{logregression, decisiontree, randomforest}_classifier.json`

The grid search parameters can be modified inside the respective config files.

#### BERT

```
python mains/bert.py -c configs/bert.json
```

The scores provided in the report were the average across runs with 5 different random seeds. The random seeds we used were [0, 1, 2, 3, 4], they were set via the config option "random_seed".

### BERTweet
For the ablation study, we ran 3 models:

```
python mains/roberta.py -c configs/roberta.json
```

For this run the option "use_special_tokens" should be set to false. You can then execute:
```
python mains/bertweet.py -c configs/bertweet.json
```
```
python mains/bertweet.py -c configs/bertweet.json
```

All runs were repeated with "random_seed" in [0, 1, 2, 3, 4].

### Additional Data

Download and extract the folder from http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip. Run the following preprocessing script:
```
python data_processing/preprocess_additional_data.py -i path-to-downloaded-folder/training.1600000.processed.noemoticon.csv -o output_folder
```

To run a model, use the same command as described above and set the config options `use_additional_data` to true. Also set `additional_negative_tweets_path` and `additional_negative_tweets_path` to the respective files generated in the output folder from the preprocessing script.

### Ensemble Learning

#### Simple Model Averaging

We used the 5 runs from the BERTweet section and gathered all class output probabilities (logged during prediction of the test set). 
Place the runs to ensemble (we sequentially averaged rs0 then rs0+rs1, then rs0+rs1+rs2, ...) inside a directory "input_directory". Run

```
python ensemble/ensemble_probabilities.py -i input_directory -o output_directory
```

Inside "output_directory" a file "ensemble_predictions.csv" will be generated.

#### Bagging

For bagging, one needs to train multiple models with the option "do_bootstrap_sampling" set to true. Then proceed as described in the simple model averaging section.

### [Section to be named] Parenthesis Rule
For this section we used either data with or without unmatched parentheses. We differentiated what data we used for training and what data for evaluation on the validation set. This in total results in 4 different possibilities per classifier. We did the following procedure for BERT and BERTweet

In a first step we trained a model with unmatched parentheses in the training data, meaning on the original labeled dataset. For this the procedure is described above. Then we trained a model without unmatched parentheses in the training data. To generate this dataset, concatenate the positive and negative tweet dataset and set the config option "remove_rule_patterns" to true. 

Both those models should be evaluated on validation data with and without unmatched parentheses. The validation data is saved in the corresponding model's checkpoint folder. Use the saved model to predict the tweets in this saved validation data, as is described at the beginning of the Reproduce Experiments section.

[] Nessi: ab da hesch alli predicted validation files, chasch echt instruction vo da namal wiiter schribe und am schluss sege dass s ganze einisch für bert und einisch für bertweet sött gmacht werde?


To reproduce the experiments, train a BERT baseline model. 

First create the tweets which have patterns of the rules removed:

```
rules/main.py -d "validation_data_path" -l "validation_labels_path" -s "save_path"
```


To compare the performance of BERT:

Test BERT on the validation_data.txt. To do this, change the test_path in the config file to path to the validation_data.txt file. Download the predictions (predictions are stored on comet.ml, access can be given on request). The checkpoint to test on should be the BERT baseline model you trained on before. This are the predictions on the unmodified tweets. 

Test BERT on the newly saved tweets where patterns of the rules are present. To do this, change the test_path in the config file to the tweet txt file you created. Download the predictions (predictions are stored on comet.ml, access can be given on request). The checkpoint to test on should be the BERT baseline model you trained on before. This are the predictions on the unmodified tweets. 


Then run the main file with the corresponding predictions from BERT to get the accuracy and the confusion matrix of bert and the rule based predictions: 

```
rules/main.py -d "validation_data_path" -l "validation_labels_path" -b "bert_predictions_path"
```


## Resource Requirements

All experiments of BERT, Roberta and BERTweet were run on ETH's Leonhard cluster using an Nvidia GeForce RTX 2080 Ti GPU. The runtimes per model were about 16 hours (26 hours with additional data) with 2 CPU cores and 64 GBs of memory for BERT and BERTweet and 96 GBs of memory for RoBERTa.
