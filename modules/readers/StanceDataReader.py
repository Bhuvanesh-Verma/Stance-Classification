import json
import logging
import random
import string
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional
from urllib.request import urlopen
from zipfile import ZipFile

import torch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)

DEFAULT_WORD_TAG_DELIMITER = "###"


@DatasetReader.register("stance_data_reader")
class StanceDataReader(DatasetReader):
    """


    # Parameters

    word_tag_delimiter: `str`, optional (default=`"###"`)
        The text that separates each WORD from its TAG.
    token_delimiter: `str`, optional (default=`None`)
        The text that separates each WORD-TAG pair from the next pair. If `None`
        then the line will just be split on whitespace.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """

    def __init__(
            self,
            word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER,
            token_delimiter: str = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            tokenizer: Optional[Tokenizer] = None,
            task: int = 1,
            val: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._word_tag_delimiter = word_tag_delimiter
        self._token_delimiter = token_delimiter
        self._tokenizer = tokenizer
        self._task = task
        self.val = val

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path, split = file_path.split('@')
        if split == 'pred':
            with open(file_path, 'r') as myfile:
                data = myfile.read()
            myfile.close()
            data = json.loads(data)
            for k, v in data.items():
                true_val = v['true']
                pred_val = v['predictions']
                yield self.text_to_instance(claim_text=true_val['claimText'],
                                            claim_target=pred_val['predictedTarget'] if 'predictedTarget' in
                                                                                        pred_val else true_val[
                                                'claimTarget'],
                                            claim_sentiment=pred_val['predictedSentiment'] if 'predictedSentiment' in
                                                                                              pred_val else str(
                                                true_val['claimSentiment']),
                                            relation=str(true_val['relation']),
                                            topic_text=true_val['topicText'],
                                            topic_target=true_val['topicTarget'],
                                            topic_sentiment=str(true_val['topicSentiment']))
        else:
            data = self.get_dataset()
            data = self.get_tags(data)
            claims = data[split]['claims']
            topics = data[split]['topics']
            random.seed(5)
            random.shuffle(claims)
            for claim in self.shard_iterable(claims):
                if 'tags' in claim:
                    yield self.text_to_instance(claim_text=claim['claimCorrectedText'],
                                                claim_target=claim['claimTarget'],
                                                tags=claim['tags'],
                                                claim_sentiment=str(claim['claimSentiment']),
                                                relation=str(claim['targetsRelation']),
                                                topic_text=topics[claim['topic_id']]['topicText'],
                                                topic_target=topics[claim['topic_id']]['topicTarget'],
                                                topic_sentiment=str(topics[claim['topic_id']]['topicSentiment']))

    def text_to_instance(  # type: ignore
            self, claim_text: str, claim_target: str = None, tags: List[str] = None, claim_sentiment: str = None,
            relation: str = None, topic_text: str = None, topic_target: str = None,
            topic_sentiment: str = None,
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        fields: Dict[str, Field] = {}
        claim_text_tokens = [Token(word) for word in claim_text.split()]
        claim_text_target_tokens = self._tokenizer.tokenize('[CLS] ' + claim_text + ' [SEP] ' + claim_target + ' [SEP]')
        claim_target_topic_target_tokens = self._tokenizer.tokenize('[CLS] ' + topic_target + ' [SEP] ' + claim_target
                                                                    + ' [SEP]')

        if self._task == 1:
            fields['tokens'] = TextField(claim_text_tokens)
            fields['tags'] = SequenceLabelField(tags, TextField(claim_text_tokens)) if tags is not None else None
            words = [x.text for x in claim_text_tokens]
        elif self._task == 2:
            fields['tokens'] = TextField(claim_text_target_tokens)
            fields['labels'] = LabelField(claim_sentiment)
            words = [x.text for x in claim_text_target_tokens]
        elif self._task == 3:
            fields['tokens'] = TextField(claim_target_topic_target_tokens)
            fields['labels'] = LabelField(relation)
            words = [x.text for x in claim_target_topic_target_tokens]
        else:
            logger.log(level=20, msg="Incorrect task id. Available task ids are 1, 2 and 3.")
            exit(0)
        fields['metadata'] = MetadataField({"claimText": claim_text, "claimTarget": claim_target, "relation": relation,
                                            "claimSentiment": claim_sentiment, "topicText": topic_text,
                                            "topicTarget": topic_target, "topicSentiment": topic_sentiment,
                                            "words": words})
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"]._token_indexers = self._token_indexers  # type: ignore

    def get_tags(self, data):
        for split, dataset in data.items():
            for index, instance in enumerate(dataset['claims']):
                sentence = instance['claimCorrectedText']
                target = instance['claimTarget']
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
                tags = [str(int(t)) for t in tag]
                data[split]['claims'][index].update({'tags': tags})
        return data

    def remove_punctuation(self, s):
        # Many instances are removed because instance contain '.' or '..' between sentence at random places which is
        # difficult to tackle
        return s.translate(str.maketrans('', '', string.punctuation))

    def get_dataset(self):
        logger.log(level=20, msg="Data is being downloaded ...")
        resp = urlopen("https://research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip")
        zipfile = ZipFile(BytesIO(resp.read()))
        file = 'IBM_Debater_(R)_CS_EACL-2017.v1/claim_stance_dataset_v1.json'

        with zipfile.open(file, 'r') as myfile:
            logger.info("Reading instances from lines in file at: %s", file)
            data = myfile.read()
        zipfile.close()
        myfile.close()
        obj = json.loads(data)
        data = {'train': {'topics': defaultdict(defaultdict), 'claims': []},
                'test': {'topics': defaultdict(defaultdict), 'claims': []},
                'val': {'topics': defaultdict(defaultdict), 'claims': []}}
        for id, topic in enumerate(obj):
            split = topic['split']
            data[split]['topics'][id] = {'topicTarget': self.remove_punctuation(topic['topicTarget']),
                                         'topicText': self.remove_punctuation(topic['topicText']),
                                         'topicSentiment': topic['topicSentiment']}
            if split == 'test':
                data['val']['topics'][id] = data[split]['topics'][id]
            for i, claim in enumerate(topic['claims']):
                if claim['Compatible'] == 'no' or claim['claimSentiment'] is None or claim['claimTarget'][
                    'text'] is None:
                    continue
                if split == 'test' and i % 4 == 0 and self.val:
                    data['val']['claims'].append({'claimTarget': self.remove_punctuation(claim['claimTarget']['text']),
                                                  'targetsRelation': claim['targetsRelation'],
                                                  'claimSentiment': claim['claimSentiment'],
                                                  'claimCorrectedText': self.remove_punctuation(
                                                      claim['claimCorrectedText']),
                                                  'topic_id': id,
                                                  'stance':claim['stance']}
                                                 )
                else:
                    data[split]['claims'].append({'claimTarget': self.remove_punctuation(claim['claimTarget']['text']),
                                                  'targetsRelation': claim['targetsRelation'],
                                                  'claimSentiment': claim['claimSentiment'],
                                                  'claimCorrectedText': self.remove_punctuation(
                                                      claim['claimCorrectedText']),
                                                  'topic_id': id,
                                                  'stance':claim['stance']}
                                                 )

        return data


''' for topic in obj:
            split = topic['split']
            if self._task == 3:
                if 'topicTarget' in topic:
                    for claim in topic['claims']:
                        if claim['Compatible'] == 'no' or claim['claimSentiment'] is None or claim['claimTarget'][
                            'text'] is None:
                            continue
                        data[split].append((topic['topicTarget'], claim['claimTarget']['text'],
                                            claim['targetsRelation']))
            else:
                if 'topicTarget' in topic:
                    data[split].append((topic['topicText'], topic['topicTarget'], topic['topicSentiment']))
                for claim in topic['claims']:
                    if claim['Compatible'] == 'no' or claim['claimSentiment'] is None or claim['claimTarget'][
                        'text'] is None:
                        continue
                    data[split].append((claim['claimCorrectedText'], claim['claimTarget']['text'],
                                        claim['claimSentiment']))'''
