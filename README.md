## About

This is a project on [Stance classification of Context-Dependent
Claims](https://aclanthology.org/E17-1024.pdf) as part of 
Project Module on Mining of Opinions and Arguments at University of Potsdam.

We use dataset created by Bar-Heim et al. 2017 for same task. This dataset
can be found [here](https://research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip).
## Usage

### Data Preprocessing
We use [SequenceTaggingDatasetReader](https://docs.allennlp.org/main/api/data/dataset_readers/sequence_tagging/)
provided by [AllenNLP](https://allennlp.org) to read data but this dataset reader 
requires data to be in certain format. So we preprocess data 
before feeding it to Dataset reader.  Execute following command 
once 
```
python install -r requirements.txt
python create_dataset.py --output_dataset Data
``` 
This will install required packages and create two files `train.txt` and `test.txt` in Data directory.  



### Training
Task 1 : Extracting Target phrase from Claim Sentence  
e.g: 
Input (Claim Sentence): "violent video games can increase children's aggression"  
Output (Target): "violent video games"
```
allennlp \
train
-s experiment/target/bert_base_uncased \
-f configs/claim_target_identification_bert_base_uncased.jsonnet

```

Task 2 : Claim Sentiment Classification
In this task, for a given claim we try to find sentiment 
of claim statement towards claim target.

e.g: Input  
(Claim Sentence): "violent video games can increase 
children's aggression", (Claim target): "violent 
video games"  
Output (Sentiment): 1 (positive sentiment)

```
allennlp \
train
-s experiment/sentiment/absa \
-f configs/claim_sentiment_classification_absa2.jsonnet

```

### Evaluation

Accuracy of models can be evaluated using following 
commands:

1. Claim Target Identification
```
allennlp \
evaluate
--cuda-device 0 \
--batch-size 8 \
experiment/target/bert_base_uncased \
Data/claim_target_test.txt
```

2. Claim Sentiment Classification

```
allennlp \
evaluate
--cuda-device 0 \
--batch-size 8 \
experiment/sentiment/absa
Data/sentiment_test.txt
```

