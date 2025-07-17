# Sign-IDD-Openpose
This codebase is an implementation of the paper Sign-IDD: Iconocity Disentangled Diffusion for Sign Language Production  
https://arxiv.org/abs/2412.13609

## Requirements
- tensorboardX for logging
- rouge_score and jiwer for evaluation
- torchview for model architecture visualisation

## Datasets
- The dataset used for both training and evaluation is PHOENIX2014T
- The input skels required for training are generated using the Openpose library
- The input gloss sequences are extracted from the dataset
- All the preprocessed files for the three spilts train, test and dev are present in the Data folder

## Training 
- The yaml file for configuration is in the Configs folder
- The summary of the entire data flow starting from preprocessing upto validation can be found in the main script
- The model checkpoints and output files are in the Models folder
- The Checkpoints folder contains the weights of the trained model

## Evaluation
- The model is evaluated using various metrics like BLEU scores, ROUGE score, WER, FID, DTW, MPJPE and MPJAE the code for all of which is in the helper and eval_helpers modules
- For calculating the NLP metrics BLEU score, ROUGE score and WER, we use the backtranslator Sign-IDD SLT
- The implementation for this backtranslator can be found in the following repository: https://github.com/Rathna-1989/SLT
- The backtranslator converts the predicted skels of Sign IDD back into gloss sequences
- These predicted gloss sequences are compared with the ground truth glosses to evaluate the interpretability of the produced poses

## Model architecture visualisation
- The input and output shapes of all the different layers of the model can be seen using the torchview library plot
- The png file is available in this repository
