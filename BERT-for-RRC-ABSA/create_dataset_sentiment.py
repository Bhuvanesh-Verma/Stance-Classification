import logging
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import argparse
import os
from typing import Optional

logger = logging.getLogger(__name__)

import os
if not os.path.exists('./asc/claimsentim/'):
    os.makedirs('./asc/claimsentim/')


def get_dataset(csv_data: Optional[str]) -> None:
    """
    This method creates train, validation and test dataset. It requires a path to store created files. Optionally, you
    can provide path to local claim_stance_dataset_v1.json file which is contained in original dataset https://research
    .ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip.
    :param csv_data: local path to claim_stance_dataset_v1.json file
    :param output_dataset: path to store dataset files
    :return: None
    """

    if csv_data is None:
        logger.log(level=20, msg="Data is being downloaded ...")
        resp = urlopen("https://research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip")
        zipfile = ZipFile(BytesIO(resp.read()))
        file = 'IBM_Debater_(R)_CS_EACL-2017.v1/claim_stance_dataset_v1.csv'
        with zipfile.open(file, 'r') as myfile:
            df = pd.read_csv(myfile)
        zipfile.close()
    else:
        with open(csv_data, 'r') as myfile:
            df = pd.read_csv(myfile)
    myfile.close()

    my_df = df[
        ['split', 'claims.claimId', 'claims.claimCorrectedText', 'claims.claimTarget.text', 'claims.claimSentiment']]
    df_train = my_df[my_df.split != 'test']
    df_train['claims.claimTarget.text'] = df_train['claims.claimTarget.text'].fillna('None')
    df_train['claims.claimSentiment'] = df_train['claims.claimSentiment'].fillna(0)
    df_train.to_json('./asc/claimsentim/train.json', orient='index')

    df_test = my_df[my_df.split != 'train']
    df_test['claims.claimTarget.text'] = df_test['claims.claimTarget.text'].fillna('None')
    df_test['claims.claimSentiment'] = df_test['claims.claimSentiment'].fillna(0)

    df_val = df_test.sample(n=300)
    df_test = df_test.drop(df_val.index)
    df_test.to_json('./asc/claimsentim/test.json', orient='index')
    df_val.to_json('./asc/claimsentim/dev.json', orient='index')


if __name__ == "__main__":
    get_dataset(csv_data=None)