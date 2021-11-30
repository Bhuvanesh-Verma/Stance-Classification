import argparse
import json
import os
import sys

import torch
import logging

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

logger = logging.getLogger(__name__)
def get_dataset(json_data: str, output_dataset: str):

    if json_data is None:
        logger.log(level=20, msg="Data is being downloaded ...")
        resp = urlopen("https://research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip")
        zipfile = ZipFile(BytesIO(resp.read()))
        file = 'IBM_Debater_(R)_CS_EACL-2017.v1/claim_stance_dataset_v1.json'
        with zipfile.open(file, 'r') as myfile:
            data = myfile.read()
    else:
        with open(json_data, 'r') as myfile:
            data=myfile.read()

    # parse file
    obj = json.loads(data)

    data = {'train':[],'test':[]}
    for topic in obj:
        split = topic['split']
        if 'topicTarget' in topic:
            data[split].append((topic['topicText'],topic['topicTarget']))
        for claim in topic['claims']:
            if 'claimTarget' in claim:
                data[split].append((claim['claimCorrectedText'],claim['claimTarget']['text']))

    tags = {'train':[], 'test':[]}
    sent_tag = {'train':[], 'test':[]}
    for split, dataset in data.items():
        for sentence, target in dataset:
            tag = torch.zeros(len(sentence.split()))
            target_len = len(target.split())
            temp_target = target.replace(' ','')
            temp_sentence = sentence.replace(target,temp_target)
            sent_list = temp_sentence.split()
            if temp_target in sent_list:
                start_index = sent_list.index(temp_target)
            else:
                continue
            end_index = start_index + target_len
            for i in range(start_index,end_index):
                tag[i] = 1
            tags[split].append(tag)
            sent_tag[split].append((sentence.split(),tag))


    for split, values in sent_tag.items():
        out_file = os.path.join(output_dataset, split + '.txt')
        with open(out_file, 'a') as the_file:
            for sents, tags in values:
                line = ''
                for sent_word, tag in zip(sents,tags):
                    line = line + str(sent_word)+"###"+str(int(tag.item()))
                    if sent_word != sents[-1]:
                        line = line +'\t'
                line = line+'\n'
                the_file.write(line)

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

    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args
    get_dataset(**vars(args))