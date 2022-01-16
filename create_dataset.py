import argparse
import json
import os
import sys
import random
from typing import Optional, Dict

import torch
import logging

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

logger = logging.getLogger(__name__)


def get_claim_target_data(obj: Dict, output_dataset: str)-> None:

    data = {'train': [], 'test': []}
    for topic in obj:
        split = topic['split']
        if 'topicTarget' in topic:
            data[split].append((topic['topicText'], topic['topicTarget']))
        for claim in topic['claims']:
            if 'claimTarget' in claim:
                data[split].append((claim['claimCorrectedText'], claim['claimTarget']['text']))

    tags = {'train': [], 'test': []}
    sent_tag = {'train': [], 'test': []}
    # Following code block is to get a mask vector for claim sentence, but there are many issues
    # So what we need to do is, if sentence is : "Banana products are superior than Tamsung." and target is: "Banana
    # products", then we need a vector [1,1,0,0,0,0]. Note that target can be anywhere in sentence. So issues are
    # 1. there are some target in dataset where target in sentence ends with comma(,). eg: " bleh bleh Dracula, bleh
    # bleh", so target for this sentence would be "Dracula" not "Dracula," and current approach cannot tackle this, so
    # it skips those instances, which leads to data loss.
    # 2. if I try to remove punctuations then there are some targets which contains punctuations.
    # If possible we need an efficient way to generate mask for a substring in a string.
    for split, dataset in data.items():
        for sentence, target in dataset:
            tag = torch.zeros(len(sentence.split()))
            target_len = len(target.split())
            temp_target = target.replace(' ', '')
            temp_sentence = sentence.replace(target, temp_target)
            sent_list = temp_sentence.split()
            if temp_target in sent_list:
                start_index = sent_list.index(temp_target)
            else:
                continue
            end_index = start_index + target_len
            for i in range(start_index, end_index):
                tag[i] = 1
            tags[split].append(tag)
            sent_tag[split].append((sentence.split(), tag))

    random.seed(5)
    random.shuffle(sent_tag['test'])
    sent_tag['val'] = []
    for i in range(300):
        sent_tag['val'].append(sent_tag['test'].pop())

    for split, values in sent_tag.items():
        out_file = os.path.join(output_dataset, 'claim_target_' + split + '.txt')
        with open(out_file, 'a') as the_file:
            for sents, tags in values:
                line = ''
                for sent_word, tag in zip(sents, tags):
                    line = line + str(sent_word) + "###" + str(int(tag.item()))
                    if sent_word != sents[-1]:
                        line = line + '\t'
                line = line + '\n'
                the_file.write(line)


def get_sentiment_data(obj: Dict, output_dataset: str) -> None:
    data = {'train': [], 'test': []}
    for topic in obj:
        split = topic['split']
        for claim in topic['claims']:
            if claim['Compatible'] == 'no' or claim['claimSentiment'] is None or claim['claimTarget']['text'] is None:
                continue
            data[split].append((claim['claimCorrectedText'].strip('\n'),
                                claim['claimTarget']['text'].strip('\n'), claim['claimSentiment']))


    random.seed(5)
    random.shuffle(data['test'])
    data['val'] = []
    for i in range(300):
        data['val'].append(data['test'].pop())

    for split, values in data.items():
        out_file = os.path.join(output_dataset, 'sentiment_' + split + '.txt')
        with open(out_file, 'a') as the_file:
            for (sentence, target, sentiment) in values:
                if sentiment is None or '\n' in sentence:
                    continue
                line = sentence + '###' + target + '###' + str(sentiment) + '\n'
                the_file.write(line)


def get_dataset(json_data: Optional[str], output_dataset: str, dataset_type: int) -> None:
    """
    This method creates train, validation and test dataset. It requires a path to store created files. Optionally, you
    can provide path to local claim_stance_dataset_v1.json file which is contained in original dataset https://research
    .ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip.
    :param dataset_type: type of dataset, 0-claim target identification, 1-Sentiment towards claim
    :param json_data: local path to claim_stance_dataset_v1.json file
    :param output_dataset: path to store dataset files
    :return: None
    """

    if json_data is None:
        logger.log(level=20, msg="Data is being downloaded ...")
        resp = urlopen("https://research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip")
        zipfile = ZipFile(BytesIO(resp.read()))
        file = 'IBM_Debater_(R)_CS_EACL-2017.v1/claim_stance_dataset_v1.json'
        with zipfile.open(file, 'r') as myfile:
            data = myfile.read()
        zipfile.close()
    else:
        with open(json_data, 'r') as myfile:
            data=myfile.read()
    myfile.close()
    # parse file
    obj = json.loads(data)

    if dataset_type == 0:
        get_claim_target_data(obj, output_dataset)
    else:
        get_sentiment_data(obj, output_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to create Dataset')

    parser.add_argument(
        '--json_data',
        help='File containing data in json format',
        type=str, required=False
    )

    parser.add_argument(
        '--output_dataset',
        help='path to save created dataset',
        type=str, required=True
    )

    parser.add_argument(
        '--dataset_type',
        help='type of dataset, 0-claim target identification, 1-Sentiment towards claim',
        type=int, required=True
    )

    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args
    get_dataset(**vars(args))