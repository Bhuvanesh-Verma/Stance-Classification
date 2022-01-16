import logging
from typing import Dict, Optional

from allennlp.common.file_utils import cached_path
from allennlp.data import Tokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("sent_data_reader2")
class SentimentDatasetReader2(DatasetReader):
    """

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file.readlines():
                line = line.strip('\n')
                if not line:
                    continue

                sentence, target, sentiment = line.split('###')
                instance = self.text_to_instance(sentence,target, sentiment)
                if instance is not None:
                    yield instance

    def text_to_instance(self, sentence: str, target:str, sentiment: int = None) -> Optional[Instance]:

        input_sentence = '[CLS] ' + sentence + ' [SEP] ' + target + ' [SEP]'
        tokens = self._tokenizer.tokenize(input_sentence)
        text_field = TextField(tokens)
        fields: Dict[str, Field] = {"tokens": text_field}

        if sentiment is not None:
            sentiment = '0' if sentiment == '-1' else '1'
            fields["label"] = LabelField(sentiment)
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"].token_indexers = self._token_indexers