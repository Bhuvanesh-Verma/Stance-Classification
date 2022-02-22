## About

This is a project on [Stance classification of Context-Dependent
Claims](https://aclanthology.org/E17-1024.pdf) as part of 
Project Module on Mining of Opinions and Arguments at University of Potsdam.

We use dataset created by Bar-Heim et al. 2017 for same task. This dataset
can be found [here](https://research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip).
## Usage

### Getting Started

For successful replication of this project we have listed packages that is required to
run our code in `requirements.txt`. Execute following command to get started
```
python install -r requirements.txt
``` 

### Data Preprocessing
We create our own dataset reader which is initially inspired from [SequenceTaggingDatasetReader](https://docs.allennlp.org/main/api/data/dataset_readers/sequence_tagging/)
provided by [AllenNLP](https://allennlp.org). Data is preprocessed within this dataset reader.

### Training
Task 1 : Extracting Target phrase from Claim Sentence  
e.g: 
Input (Claim Sentence): "violent video games can increase children's aggression"  
Output (Target): "violent video games"
```
allennlp \
train
-s experiment/train/target/bert_base_uncased \
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
-s experiment/train/sentiment/absa \
-f configs/claim_sentiment_classification_absa.jsonnet

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
experiment/train/target/bert_base_uncased \
_@test
```

2. Claim Sentiment Classification

```
allennlp \
evaluate
--cuda-device 0 \
--batch-size 8 \
experiment/train/sentiment/absa
_@test
```

### Prediction


1. Claim Target Identification
```
allennlp \
predict-stance
--cuda-device 0 \
--batch-size 8 \
--output-file experiment/prediction/target/prediction.txt \
--use-dataset-reader \
--silent \
experiment/train/target/bert_base_uncased \
_@test
```

2. Claim Sentiment Classification

```
allennlp \
predict-stance
--cuda-device 0 \
--batch-size 8 \
--output-file experiment/prediction/sentiment/prediction.txt \
--use-dataset-reader \
--silent \
experiment/train/sentiment/absa \
_@test
```

