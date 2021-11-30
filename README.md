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
python main.py \
train \
-s experiments \
-f configs/claim_target_identification.jsonnet

```