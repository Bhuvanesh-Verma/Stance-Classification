import json
import random
from io import BytesIO
from typing import Dict, List, Optional
import logging
from zipfile import ZipFile

import torch
from allennlp.data.tokenizers.tokenizer import Tokenizer
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from urllib.request import urlopen

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
    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path, split = file_path.split('@')
        if split=='pred':
            with open(file_path, 'r') as myfile:
                for line in myfile.readlines():
                    line = line.strip('\n')
                    if not line:
                        continue
                    sentence, target, sentiment = line.split('###')
                    yield self.text_to_instance(sentence=sentence, target=target, sentiment=sentiment)
        else:
            data = self.get_dataset()
            sent_tag = self.get_tags(data)
            random.seed(5)
            random.shuffle(sent_tag['test'])
            sent_tag['val'] = []
            for i in range(300):
                sent_tag['val'].append(sent_tag['test'].pop())
            for sentence, target, sentiment, tags in self.shard_iterable(sent_tag[split]):
                if self._task == 1:
                    tags = [str(int(tag)) for tag in tags]
                    yield self.text_to_instance(sentence=sentence, tags=tags, sentiment=sentiment)
                elif self._task == 2:
                    yield self.text_to_instance(sentence=sentence, target=target, sentiment=sentiment)
                elif self._task == 3:
                    pass
                else:
                    pass

    def text_to_instance(  # type: ignore
        self, sentence: str, target: str = None, tags: List[str] = None, sentiment: int = None
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        fields: Dict[str, Field] = {}
        if self._task == 1:
            tokens = [Token(word) for word in sentence.split()]
        if self._task == 2:
            sentence = '[CLS] ' + sentence + ' [SEP] ' + target + ' [SEP]'
            tokens = self._tokenizer.tokenize(sentence)
        sequence = TextField(tokens)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence)
        if sentiment is not None:
            sentiment = str(sentiment) #'0' if str(sentiment) == '-1' else '1'
            fields["sentiment"] = LabelField(sentiment)
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"]._token_indexers = self._token_indexers  # type: ignore

    def get_tags(self, data):
        sent_tag = {'train': [], 'val': [], 'test': []}
        for split, dataset in data.items():
            for sentence, target, sentiment in dataset:
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
                sent_tag[split].append((sentence, target, sentiment, tag))
        return sent_tag

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
        data = {'train': [], 'test': []}
        for topic in obj:
            split = topic['split']

            if 'topicTarget' in topic:
                data[split].append((topic['topicText'], topic['topicTarget'], topic['topicSentiment']))
            for claim in topic['claims']:
                if claim['Compatible'] == 'no' or claim['claimSentiment'] is None or claim['claimTarget'][
                    'text'] is None:
                    continue
                data[split].append((claim['claimCorrectedText'], claim['claimTarget']['text'],
                                    claim['claimSentiment']))

        return data
