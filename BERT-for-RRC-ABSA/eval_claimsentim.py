import argparse
import json
import os
import sklearn.metrics
import numpy as np
import sys

import json
import os
import sklearn.metrics
import numpy as np


def evaluate(task, bert, domain, runs=10):
    scores = []
    for run in range(1, runs + 1):
        DATA_DIR = os.path.join(task, domain)
        OUTPUT_DIR = os.path.join("run", bert + "_" + task, domain, str(run))
        if os.path.exists(os.path.join(OUTPUT_DIR, "predictions.json")):
            if "asc" in task:
                with open(os.path.join(OUTPUT_DIR, "predictions.json")) as f:
                    results = json.load(f)
                y_true = results['label_ids']
                y_pred = [np.argmax(logit) for logit in results['logits']]
                p_macro, r_macro, f_macro, _ = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                                               average='macro')
                f_macro = 2 * p_macro * r_macro / (p_macro + r_macro)
                scores.append([100 * sklearn.metrics.accuracy_score(y_true, y_pred), 100 * f_macro])
            else:
                raise Exception("unknown task")
    scores = np.array(scores)
    m = scores.mean(axis=0)

    if len(scores.shape) > 1:
        for iz, score in enumerate(m):
            print(task, ":", bert, domain, "metric", iz, round(score, 2))
    else:
        print(task, ":", bert, domain, round(m, 2))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to evaluate the finetuned model')

    parser.add_argument(
        '--task',
        help='The task for evaluation (asc, ae)',
        type=str, required=True
    )

    parser.add_argument(
        '--bert',
        help='The name of BERT model used',
        type=str, required=True
    )
    parser.add_argument(
        '--domain',
        help='The name of the dataset folder',
        type=str, required=True
    )
    parser.add_argument(
        '--runs',
        help='The number of runs for finetuning',
        type=int, required=True
    )

    args = parser.parse_args()
    evaluate(**vars(args))
