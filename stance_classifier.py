import json
import sys

import argparse

def get_predictions(pred_file: str):
    with open(pred_file, 'r') as myfile:
        data = myfile.read()
    myfile.close()
    return json.loads(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to classify stance')

    parser.add_argument(
        '--pred_rel_file',
        help='Prediction file with predicted relation',
        type=str, required=False
    )

    parser.add_argument(
        '--pred_sent_file',
        help='Prediction file with predicted sentiment',
        type=str, required=False
    )

    args, remaining_args = parser.parse_known_args()

    rel_data = get_predictions(args.pred_rel_file)
    sent_data = get_predictions(args.pred_sent_file)
    topics = {}
    correct = 0
    for k, v in sent_data.items():
        if v['true']['topic']:
            topics[v['true']['topic_id']] = v['true']['topicSentiment']
    for k, v in rel_data.items():
        if not v['true']['topic']:
            claim_sentiment = int(v['true']['predictedClaimSentiment'])
            target_sentiment = int(topics[v['true']['topic_id']])
            relation = int(v['predictions']['predictedRelation'])
            stance = claim_sentiment * target_sentiment * relation

            if (stance == 1 and v['true']['stance'] == 'PRO') or (stance == -1 and v['true']['stance'] == 'CON'):
                correct += 1
    total = len(rel_data.keys())
    print(f'Accuracy: {correct*100/total}')
