## About

This is a project on [Stance classification of Context-Dependent
Claims](https://aclanthology.org/E17-1024.pdf) as part of 
Project Module on Mining of Opinions and Arguments at University of Potsdam.

We use dataset created by Bar-Heim et al. 2017 for same task. This dataset
can be found [here](https://research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip).
## Usage

### Getting Started

For successful replication of this project we have listed packages that are required to
run our code in `requirements.txt`. Execute following commands to get started
```
conda create -n testenv python=3.8
conda activate testenv
pip install -r requirements.txt
``` 

### Data Preprocessing
We create our own dataset reader which is initially inspired from [SequenceTaggingDatasetReader](https://docs.allennlp.org/main/api/data/dataset_readers/sequence_tagging/)
provided by [AllenNLP](https://allennlp.org). Data is preprocessed within this [dataset reader](modules/readers/StanceDataReader.py).

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
Plots from model training and hyper parameter training can be 
found [here](experiment/train/target).
<hr>
Task 2 : Claim Sentiment Classification

In this task, for a given claim we try to find sentiment 
of claim statement towards claim target.

e.g: Input  
(Claim Sentence): "violent video games can increase 
children's aggression"  
(Claim target): "violent 
video games"  
Output (Sentiment): 1 (positive sentiment)

```
allennlp \
train
-s experiment/train/sentiment/absa \
-f configs/claim_sentiment_classification_absa.jsonnet

```
Plots for task 2 training can be found [here](experiment/train/sentiment).
<hr>
Task 3 : Contrast Classification (Semantic Similarity Classification)  

In this task, relation between topic and claim target is
classified. Originally, task is to classify 
targets as consistent or contrastive. We tried to find 
semantic similarity between targets instead.

e.g: Input  
(Topic Target): "the sale of violent video games to minors",  
(Claim target): "violent 
video games"  
Output (relation or similarity): 1 (similar or consistent)

```
allennlp \
train
-s experiment/train/contrast/bilstm \
-f configs/contrast_classification_bilstm.jsonnet

```
Plots for task 3 training can be found [here](experiment/train/contrast).


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
`experiment/train/target/bert_base_uncased` is path to trained model.   
We can also evaluate model by using full test data by setting
`val` flag off in dataset reader. We override config value
of `val` flag in command below to achieve this.

```
allennlp \
evaluate
--cuda-device 0 \
--batch-size 8 \
experiment/train/target/bert_base_uncased \
_@test \
-o "{\"dataset_reader.val\":false}"
```

2. Claim Sentiment Classification

```
allennlp \
evaluate
--cuda-device 0 \
--batch-size 8 \
experiment/train/sentiment/lpt \
_@test
```
To evaluate using predicted target replace _@test with 
PATH_TO_TASK1_PREDICTION@pred. We provide prediction files
from our trained model (see below in Prediction).

3. Contrast Classification or Semantic Similarity Classification

```
allennlp \
evaluate
--cuda-device 0 \
--batch-size 32 \
experiment/train/contrast/bilstm \
_@test
```
To evaluate using predicted target replace _@test with 
PATH_TO_TASK2_PREDICTION@pred. Prediction of task 2 contains
predicted tags from task1. 

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

Prediction from our best model for task 1 can be found [here](experiment/prediction/target/prediction.txt). 

2. Claim Sentiment Classification

Prediction from gold targets
```
allennlp \
predict-stance
--cuda-device 0 \
--batch-size 8 \
--output-file experiment/prediction/sentiment/lpt_prediction.txt \
--use-dataset-reader \
--silent \
experiment/train/sentiment/lpt \
_@test
```
Prediction from our best model for task 2 on gold targets can be found [here](experiment/prediction/sentiment/lpt_prediction.txt).

Prediction from predicted targets
```
allennlp \
predict-stance
--cuda-device 0 \
--batch-size 8 \
--output-file experiment/prediction/sentiment/lpt_prediction_from_pred.txt \
--use-dataset-reader \
--silent \
experiment/train/sentiment/lpt \
experiment/prediction/target/prediction.txt@pred
```
Prediction for task 2 on predicted targets can be found [here](experiment/prediction/sentiment/lpt_prediction_from_pred.txt).

3. Contrast Classification or Semantic Similarity Classification

Prediction from gold targets
```
allennlp \
predict-stance
--cuda-device 0 \
--batch-size 32 \
--output-file experiment/prediction/contrast/prediction.txt \
--use-dataset-reader \
--silent \
experiment/train/contrast/bilstm \
_@test \
```
Prediction from our best model for task 3 on gold targets can be found [here](experiment/prediction/contrast/prediction.txt).

Prediction from predicted targets

```
allennlp \
predict-stance
--cuda-device 0 \
--batch-size 32 \
--output-file experiment/prediction/contrast/prediction_from_pred.txt \
--use-dataset-reader \
--silent \
experiment/train/contrast/bilstm \
experiment/prediction/sentiment/lpt_prediction_from_pred.txt@pred \
```
Prediction for task 3 on predicted targets can be found [here](experiment/prediction/contrast/prediction_from_pred.txt).


### Stance Classification
This will calculate the macro averaged accuracy of our stance classification model. It requires 
prediction files from task 2 and task 3.

We test our model in two ways.
1. Gold Targets  
First we predict [sentiment towards target](experiment/prediction/sentiment/lpt_prediction.txt)
using `gold` targets. Then we predict [relation](experiment/prediction/contrast/prediction.txt) between topic 
and claim targets using `gold` targets. Finally, we use these to prediction files to evaluate our model.

```
python stance_classifier.py \
--pred_rel_file \
experiment/prediction/contrast/prediction.txt \
--pred_sent_file \
experiment/prediction/sentiment/lpt_prediction.txt
```

2. Predicted Targets  
First we predict the [claim targets](experiment/prediction/target/prediction.txt) and then using the predicted 
claim target, we predict [sentiment towards predicted target](experiment/prediction/sentiment/lpt_prediction_from_pred.txt)
and then we use the predicted target to [predict relation](experiment/prediction/contrast/prediction_from_pred.txt).
These prediction files are used to evaluate model.

```
python stance_classifier.py \
--pred_rel_file \
experiment/prediction/contrast/prediction_from_pred.txt \
--pred_sent_file \
experiment/prediction/sentiment/lpt_prediction_from_pred.txt
```
| Type              | Pro Accuracy | Con Accuracy | Macro Averaged Accuracy |
|-------------------|--------------|--------------|-------------------------|
| Gold Targets      | 70.36        | 71.85        | 71.1                    |
| Predicted Targets | 70.12        | 70.87        | 70.5                    |

### Project IBM Debater

We tested our model against Pro/Con stance classification
model of Project IBM debater. Their model can predict Pro,
Con and neutral for a sentence-topic pair in range -1 to 1, however there is no specific 
boundary provided. We choose three different thresholds to test their model.
`Threshold=0.33` means that if model output is greater than 0.33 then sentence is 
labelled as `Pro` and if it is less than -0.33 then `Con`. 



| Threshold | Pro Accuracy | Con Accuracy | Macro Averaged Accuracy |
|-----------|--------------|--------------|-------------------------|
| 0         | 81.53        | 73.79        | 77.66                   |
| 0.33      | 73.87        | 66.5         | 70.19                   |
| 0.5       | 70.42        | 63.59        | 67.01                   |
                

