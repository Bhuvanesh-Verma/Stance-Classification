import json
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from debater_python_api.api.debater_api import DebaterApi

debater_api = DebaterApi('6bfc73e9be5066f1f1555a4fd0aaebe2L05')
pro_con_client = debater_api.get_pro_con_client()

resp = urlopen("https://research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip")
zipfile = ZipFile(BytesIO(resp.read()))
file = 'IBM_Debater_(R)_CS_EACL-2017.v1/claim_stance_dataset_v1.json'

with open('experiment/prediction/contrast/prediction_from_pred.txt', 'r') as myfile:
    data = myfile.read()
myfile.close()
obj = json.loads(data)
sentence_topic_dicts = []
true_stance = []
for k,v in obj.items():
    if not v['true']['topic']:
        topic = v['true']['topicText']
        claim = v['true']['claimCorrectedText']
        stance = v['true']['stance']
        sentence_topic_dicts.append({'sentence': claim, 'topic':topic})
        true_stance.append(stance)
scores = pro_con_client.run(sentence_topic_dicts)

count_pro = 0
total_pro = 0
count_con = 0
total_con = 0
for pred_, true in zip(scores, true_stance):
    if pred_ > 0.33:
        pred = 'PRO'
    elif pred_ < -0.33:
        pred = 'CON'
    else:
        if true == 'PRO':
            total_pro += 1
        if true == 'CON':
            total_con += 1
        continue
    if true == 'PRO':
        total_pro += 1
        if true == pred:
            count_pro += 1
    if true == 'CON':
        total_con += 1
        if true == pred:
            count_con += 1

pro_acc = count_pro/total_pro
con_acc = count_con/total_con

acc = 0.5*pro_acc + 0.5*con_acc
print(f'Accuracy: {acc*100}')
