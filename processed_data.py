

import os
import torch
from tqdm.auto import tqdm
from utils.data_utils import *


if __name__ == '__main__': 

    path = './data/raw_data'
    files = [x for x in os.listdir(path) if x.endswith('ent')]
    
    # s3
    # session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    # s3_client = session.client("s3")
    
    
    for file in tqdm(files, desc='Processing and Uploading to S3'):
        try: 
            file_name = os.path.join(path, file)
            data = construct_data_from_pdb(file_name, describe=True)
            
            # save into the processed data
            dataname = data.name[0]
            torch.save(data, f'./data/processed_data/{dataname}.pt')

            object_name = f'AllProteinComplexes/processed_data/{dataname}.pt'
            # upload_file_to_s3(s3_client, file_name, BUCKET, object_name)
        except: 
            print(f'Error encountered at {file}..')
            continue

    print('SUCCESS!!')