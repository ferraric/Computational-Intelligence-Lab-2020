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
mv .../downloaded-data-folder ~/data

# --> if on Leonhard: install your python virtual environment into Computational-Intelligence-Lab-2020/venv
python3 -m venv /cluster/home/.../Computational-Intelligence-Lab-2020/venv
source ./init_leonhard.sh

# --> if local: 
pip install -r requirements.txt

 ```  
We use pre-commit hooks to format our code to comply with black and pep8. If you want to contribute execute: 
```pre-commit install```
 
