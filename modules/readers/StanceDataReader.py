import json
import logging
import random
import string
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple
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
    Generalised Dataset reader for all three sub tasks of Stance classification.
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
        """
        This method downloads dataset, convert it into required format for each subtask.
        :param file_path: it is usually in the format '_@split' or 'PATH@split'
        :return: None
        """
        file_path, split = file_path.split('@')
        if split == 'pred':
            # This blocked is accessed only during the evaluation and prediction.
            with open(file_path, 'r') as myfile:
                data = myfile.read()
            myfile.close()
            data = json.loads(data)
            for k, v in data.items():
                true_val = v['true']
                pred_val = v['predictions']
                metadata = true_val
                if true_val['topic']:

                    if self._task == 2 and pred_val['predictedTarget'] is not None:
                        metadata.update({'predictedTopicTarget':pred_val['predictedTarget']})
                        yield self.text_to_instance(sentences=(true_val['topicText'], pred_val['predictedTarget']),
                                                    label=str(true_val['topicSentiment']), metadata=metadata)
                    elif self._task == 3 and pred_val['predictedSentiment'] is not None:
                        metadata.update({'predictedTopicSentiment': pred_val['predictedSentiment']})
                else:
                    if self._task == 2 and pred_val['predictedTarget'] is not None:
                        metadata.update({'predictedClaimTarget':pred_val['predictedTarget']})
                        yield self.text_to_instance(sentences=(true_val['claimCorrectedText'], pred_val['predictedTarget']),
                                                    label=str(true_val['claimSentiment']), metadata=metadata)
                    elif self._task == 3 and pred_val['predictedSentiment'] is not None:
                        metadata.update({'predictedClaimSentiment': pred_val['predictedSentiment']})
                        yield self.text_to_instance(sentences=(true_val['topicTarget'], true_val['predictedClaimTarget']),
                                                    label=str(true_val['targetsRelation']), metadata=metadata)
        else:
            data = self.get_dataset()
            claims = data[split]['claims']
            topics = data[split]['topics']
            random.seed(5)
            random.shuffle(claims)
            for claim in self.shard_iterable(claims):
                metadata = claim
                metadata.update(topics[claim['topic_id']])
                metadata.update({'topic':False})
                if self._task == 1 and claim['c_tags'] is not None:
                    yield self.text_to_instance(sentences=(claim['claimCorrectedText'],claim['claimTarget']),
                                                label=claim['c_tags'], metadata=metadata)
                elif self._task == 2 and claim['claimSentiment'] is not None:
                    yield self.text_to_instance(sentences=(claim['claimCorrectedText'], claim['claimTarget']),
                                                label=str(claim['claimSentiment']), metadata=metadata)
                elif self._task == 3 and claim['targetsRelation'] is not None:
                    yield self.text_to_instance(sentences=(topics[claim['topic_id']]['topicTarget'], claim['claimTarget']),
                                                label=str(claim['targetsRelation']), metadata=metadata)

            for id,topic in self.shard_iterable(topics.items()):
                metadata = topic
                metadata.update({'topic_id':id,'topic':True})
                if self._task == 1 and topic['t_tags'] is not None:
                    yield self.text_to_instance(sentences=(topic['topicText'],topic['topicTarget']),
                                                label=topic['t_tags'], metadata=metadata)
                elif self._task == 2 and topic['topicSentiment'] is not None:
                    yield self.text_to_instance(sentences=(topic['topicText'],topic['topicTarget']),
                                                label=str(topic['topicSentiment']), metadata=metadata)


    def text_to_instance(  # type: ignore
            self, sentences: Tuple[str,str], label, metadata
    ) -> Instance:
        """
        It converts data from text to Instance format. To generalise this method for all tasks, we take sentences which
        is tuple of strings as in each task we require two sentences as input. It then converts these sentences to
        required format as per the model used.
        """

        fields: Dict[str, Field] = {}
        if self._task == 1:
            tokens = [Token(word) for word in sentences[0].split()]
            fields['tokens'] = TextField(tokens)
            fields['tags'] = SequenceLabelField(label, fields['tokens']) if label is not None else None
        elif self._task == 2 or self._task == 3:
            tokens = self._tokenizer.tokenize('[CLS] ' + sentences[0] + ' [SEP] ' + sentences[1] + ' [SEP]')
            s = [x.text for x in tokens].index('[SEP]')
            for i in range(s,len(tokens)):
                tokens[i].type_id=1
            fields['tokens'] = TextField(tokens)
            fields['labels'] = LabelField(label)
        else:
            logger.log(level=20, msg="Incorrect task id. Available task ids are 1, 2 and 3.")
            exit(0)

        metadata.update({'words': [x.text for x in tokens]})
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"]._token_indexers = self._token_indexers  # type: ignore

    def get_tags(self, data)->Dict:
        """
        This method retrieve binary tags for claim and topic sentence.
        :param data: dictionary containing data divided by split
        :return: data containing tags for claims and targets of all splits.
        """
        for split, dataset in data.items():
            for index, instance in enumerate(dataset['claims']):
                sentence = instance['claimCorrectedText']
                target = instance['claimTarget']
                tags = self.create_tags(sentence=sentence,target=target)
                data[split]['claims'][index].update({'c_tags': tags})
            for id, values in dataset['topics'].items():
                sentence = values['topicText']
                target = values['topicTarget']
                tags = self.create_tags(sentence=sentence,target=target)
                data[split]['topics'][id].update({'t_tags': tags})
        return data

    def create_tags(self, sentence, target):
        """
        For a given sentence and target, this method creates a list of 1s and 0s as tags. Portion of the sentence
        containing target is tagged as 1 and rest with 0.
        :param sentence: string sentence
        :param target: string phrase as target
        :return: list with 1s and 0s representing tags for the sentence
        """
        tag = torch.zeros(len(sentence.split()))
        target_len = len(target.split())
        temp_target = target.replace(' ', '')
        temp_sentence = sentence.replace(target, temp_target)
        sent_list = temp_sentence.split()
        if temp_target in sent_list:
            start_index = sent_list.index(temp_target)
        else:
            return None
        end_index = start_index + target_len
        for i in range(start_index, end_index):
            tag[i] = 1
        tags = [str(int(t)) for t in tag]
        return tags

    def remove_punctuation(self, s)-> str:
        # Many instances are removed because instance contain '.' or '..' between sentence at random places which is
        # difficult to tackle
        return s.translate(str.maketrans('', '', string.punctuation))

    def get_dataset(self)->Dict:
        """
        This method downloads IBM_Debater_(R)_CS_EACL-2017 dataset and parse the claim_stance_dataset_v1.json data file.
        It converts data from data file into a dictionary which contains splits and for each split there are topics and
         claims. Then for each claim and topic sentence binary tags are created for performing sequence tagging task.
        :return: dictionary containing data with respect to the splits.
        """
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
        data = self.get_tags(data)

        return data