import json

import argparse

from typing import Dict

'''
In this script we calculate stance classification using the formula derived by Bar-Heim et al. 2017. This script takes
two arguments, sentiment and relation prediction file. It then use predicted sentiment and relation to calculate stance.
Finally, it presents accuracy for Pro and Con labels as well as overall macro average accuracy.
'''

def get_predictions(pred_file: str) -> Dict[str, Dict]:
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
    correct_pro = 0
    correct_con = 0
    total_pro = 0
    total_con = 0
    for k, v in rel_data.items():
        if not v['true']['topic']:
            claim_sentiment = int(sent_data[k]['predictions']['predictedSentiment'])
            target_sentiment = int(v['true']['topicSentiment'])
            relation = int(v['predictions']['predictedRelation'])
            stance = claim_sentiment * target_sentiment * relation

            if v['true']['stance'] == 'PRO':
                total_pro += 1
                if stance == 1:
                    correct_pro += 1
            if v['true']['stance'] == 'CON':
                total_con += 1
                if stance == -1:
                    correct_con += 1
    total = len(rel_data.keys())
    pro_accuracy = correct_pro/total_pro
    con_accuracy = correct_con/total_con
    print(f'Pro Accuracy:{pro_accuracy} ')
    print(f'Con Accuracy:{con_accuracy} ')
    print(f'macro averaged Accuracy: {(pro_accuracy*0.5 + con_accuracy*0.5)*100}')
