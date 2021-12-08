
import json
from typing import List, Dict, Any, Optional
from flatten_dict import flatten
import os
import copy
from allennlp import commands
import sys

from allennlp.common.params import parse_overrides


def maybe_convert_to_numeric(v: Any):
    try:
        return int(v)  # this will execute if string v is integer
    except ValueError:
        pass
    try:
        return float(v)  # this will execute if string v is float
    except ValueError:
        pass

    return v


def integrate_into_dict(d: Dict, key: List[str], value: Any):
    if len(key) == 1:
        d[key[0]] = value
    else:
        d[key[0]] = integrate_into_dict(d=d.get(key[0], {}), key=key[1:], value=value)
    return d


def integrate_arguments_and_run_allennlp():

    args = sys.argv
    overrides = {}

    keys = ["--model", "--trainer", "--dataset_reader", "--data_loader"]
    print(f'original arguments:{args}')
    rem_keys = []
    i = 0
    while i < len(sys.argv):
        flag = False
        # Interpret all arguments starting with '-o' or '--overrides' as overrides for allennlp command and delete
        # from sys arguments
        if sys.argv[i] == '-o' or sys.argv[i] == '--overrides':
            overrides.update(parse_overrides(sys.argv[i + 1]))
            del sys.argv[i]
            del sys.argv[i]
            continue
        # Save all the arguments passed by the sweep and delete from sys arguments
        for prefix in keys:
            if i == len(sys.argv):
                break
            if sys.argv[i].startswith(prefix):
                rem_keys.append(sys.argv[i])
                del sys.argv[i]
                flag = True

        if not flag:
            i += 1

    for arg in rem_keys:
        arg = arg[2:]
        arg_k, arg_v = arg.split("=")
        arg_v = maybe_convert_to_numeric(arg_v)
        overrides = integrate_into_dict(overrides, arg_k.split("."), arg_v)

    sys.argv.append("-o")
    overrides = flatten(overrides, reducer='dot')
    sys.argv.append(json.dumps(overrides))
    print(f'updated arguments:{sys.argv}')
    commands.main()


if __name__ == '__main__':
    integrate_arguments_and_run_allennlp()