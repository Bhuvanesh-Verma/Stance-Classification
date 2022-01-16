# BERT for detecting sentiment towards the topic of the claim 
We adapted the code from [BERT-for-RCC-ABSA](https://github.com/howardhsu/BERT-for-RRC-ABSA) for our task.

**Task description**:

**ASC**: given a claim ("Violent video games increase the violent tendencies among youth") and its target ("Violent video games"), detect the detect the sentiment of the claim towards topic (negative).

The code is tested on Ubuntu 18.04 with Python 3.6.8(Anaconda), PyTorch 1.0.1 and [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT) 0.4. 

## Fine-tuning Setup

1) The pre-trained model is saved in the folder ```pt_model```.


2) The model weights that were used for fine-tuning can be found [here](https://drive.google.com/file/d/1io-_zVW3sE6AbKgHZND4Snwh-wi32L4K/view?usp=sharing).

TODO: fine-tune on the model found [here](https://drive.google.com/file/d/1TYk7zOoVEO8Isa6iP0cNtdDFAUlpnTyz/view?usp=sharing). 
And other BERT weights (e.g., bert-base, BERT-DK ([laptop](https://drive.google.com/file/d/1TRjvi9g3ex7FrS2ospUQvF58b11z0sw7/view?usp=sharing), [restaurant](https://drive.google.com/file/d/1nS8FsHB2d-s-ue5sDaWMnc5s2U1FlcMT/view?usp=sharing)) in our paper). 
Make sure to add an entry into ```src/modelconfig.py```.

### Create a dataset

4) Run the script ```create_dataset_sentiment.py``` to create the dataset fot the task. (code needs revision)

### Training 
5) All the logs for training and testing are stored in ```run```.
6) Fire a fine-tuning from a BERT weight, e.g.
```
cd script
bash run_absa.sh asc laptop_pt claimsentim pt_asc 10 0
```
Here ```asc``` is the task to run, ```laptop_pt``` is the post-trained weights for laptop, ```claimsentim``` is the dataset we want to fine-tune on, ```pt_asc``` is the fine-tuned folder in ```run/```, ```10``` means run 10 times and ```0``` means use gpu-0.

TODO: try fine-tuning on the task of aspect extraction (need to preprocess the dataset)
```
bash run_absa.sh ae laptop_pt claimsentim pt_ae 10 0
```
### Evaluation

**ASC**: run the script (returns accuracy and f_macro scores)

```
eval_claimsentim.py /
--task asc --bert pt --domain claimsentim --runs 10
```

**AE**: place official evaluation .jar files as ```eval/A.jar``` and ```eval/eval.jar```.
place testing xml files as (the step 4 of [this](https://github.com/howardhsu/DE-CNN) has a similar setup)
```
ae/official_data/Laptops_Test_Gold.xml
ae/official_data/Laptops_Test_Data_PhaseA.xml
ae/official_data/EN_REST_SB1_TEST.xml.gold
ae/official_data/EN_REST_SB1_TEST.xml.A
```
