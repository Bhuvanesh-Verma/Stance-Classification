"""
The `predict` subcommand allows you to make bulk JSON-to-JSON
or dataset to JSON predictions using a trained model and its
[`Predictor`](../predictors/predictor.md#predictor) wrapper.
"""
import argparse
import json
import sys
from typing import List, Iterator, Optional

from allennlp.commands import Predict
from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.data import Instance
from allennlp.data.dataset_readers import MultiTaskDatasetReader
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from overrides import overrides


@Subcommand.register("predict-stance")
class PredictTarget(Predict):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = super().add_subparser(parser)

        subparser.set_defaults(func=_predict)

        return subparser


def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )

    predictor_args = args.predictor_args.strip()
    if len(predictor_args) <= 0:
        predictor_args = {}
    else:
        import json

        predictor_args = json.loads(predictor_args)

    return Predictor.from_archive(
        archive,
        args.predictor,
        dataset_reader_to_load=args.dataset_reader_choice,
        extra_args=predictor_args,
    )


class _PredictManager:
    def __init__(
            self,
            predictor: Predictor,
            input_file: str,
            output_file: Optional[str],
            batch_size: int,
            print_to_console: bool,
            has_dataset_reader: bool,
            multitask_head: Optional[str] = None,
    ) -> None:
        self._predictor = predictor
        self._input_file = input_file
        self._output_file = None if output_file is None else open(output_file, "w")
        self._batch_size = batch_size
        self._print_to_console = print_to_console
        self._dataset_reader = None if not has_dataset_reader else predictor._dataset_reader

        self._multitask_head = multitask_head
        if self._multitask_head is not None:
            if self._dataset_reader is None:
                raise ConfigurationError(
                    "You must use a dataset reader when using --multitask-head."
                )
            if not isinstance(self._dataset_reader, MultiTaskDatasetReader):
                raise ConfigurationError(
                    "--multitask-head only works with a multitask dataset reader."
                )
        if (
                isinstance(self._dataset_reader, MultiTaskDatasetReader)
                and self._multitask_head is None
        ):
            raise ConfigurationError(
                "You must specify --multitask-head when using a multitask dataset reader."
            )

    def _predict_json(self, batch_data: List[JsonDict]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_json(batch_data[0])]
        else:
            results = self._predictor.predict_batch_json(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _predict_instances(self, batch_data: List[Instance]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0])]
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            if self._dataset_reader._task == 2:
                label = output['label']
                result = {'predictedSentiment': label}
                yield result
            if self._dataset_reader._task == 1:
                pred_target = [word for word, tag in zip(output['words'], output['tags']) if tag == '1']
                result = {'predictedTarget': ' '.join(pred_target)}
                yield result
            if self._dataset_reader._task == 3:
                label = output['label']
                result = {'predictedRelation': label}
                yield result

    def _maybe_print_to_console_and_file(
            self, index: int, prediction: str, model_input: Instance = None
    ) -> None:
        if self._print_to_console:
            if model_input is not None:
                print(f"input {index}: ", str(model_input))
            print("prediction: ", str(prediction))
        if self._output_file is not None:
            self._output_file.write(prediction)

    def _get_json_data(self) -> Iterator[JsonDict]:
        if self._input_file == "-":
            for line in sys.stdin:
                if not line.isspace():
                    yield self._predictor.load_line(line)
        else:
            input_file = cached_path(self._input_file)
            with open(input_file, "r") as file_input:
                for line in file_input:
                    if not line.isspace():
                        yield self._predictor.load_line(line)

    def _get_instance_data(self) -> Iterator[Instance]:
        if self._input_file == "-":
            raise ConfigurationError("stdin is not an option when using a DatasetReader.")
        elif self._dataset_reader is None:
            raise ConfigurationError("To generate instances directly, pass a DatasetReader.")
        else:
            if isinstance(self._dataset_reader, MultiTaskDatasetReader):
                assert (
                        self._multitask_head is not None
                )  # This is properly checked by the constructor.
                yield from self._dataset_reader.read(
                    self._input_file, force_task=self._multitask_head
                )
            else:
                yield from self._dataset_reader.read(self._input_file)

    def run(self) -> None:
        has_reader = self._dataset_reader is not None
        index = 0
        output = {}
        if has_reader:
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    output[index] = {'predictions': result, 'true': dict(model_input_instance['metadata'])}
                    index = index + 1
            self._output_file.write(json.dumps(output, indent=2))
            self._output_file.close()
        else:
            for batch_json in lazy_groups_of(self._get_json_data(), self._batch_size):
                for model_input_json, result in zip(batch_json, self._predict_json(batch_json)):
                    self._maybe_print_to_console_and_file(
                        index, result, json.dumps(model_input_json)
                    )
                    index = index + 1

        if self._output_file is not None:
            self._output_file.close()


def _predict(args: argparse.Namespace) -> None:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    predictor = _get_predictor(args)

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    manager = _PredictManager(
        predictor,
        args.input_file,
        args.output_file,
        args.batch_size,
        not args.silent,
        args.use_dataset_reader,
        args.multitask_head,
    )
    manager.run()
