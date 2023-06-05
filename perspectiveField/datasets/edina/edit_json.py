import albumentations as A
import json
import numpy as np
import random
import cv2
import os,git,json
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from panocam import PanoCam
from perspective2d.utils import draw_latitude_field, draw_up_field, general_vfov
from sklearn.preprocessing import normalize
import shutil


def assign_info(phase, dataset_dicts, dataset, root):
    description = f"""{dataset} Height {phase} Dataset."""
    repo = git.Repo(search_parent_directories=True)
    git_hexsha = repo.head.object.hexsha
    date_created = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    rtn = {
        'info': {
            'description': description,
            'git_hexsha': git_hexsha,
            'date_created': date_created,
            'root': root,
            'phase': phase,
            'dataset': dataset,
        },
    }
    dump_file_summary = {}
    dump_file_summary.update(rtn)
    dump_file_summary['data'] = dataset_dicts
    return dump_file_summary


if __name__ == '__main__':
    save_f = '/nfs/turbo/fouheyTemp/msticha/datasets/edina_test'
    input_json_f = 'edina_test.json'
    output_json_f = 'edina_test_edited.json'
    with open(input_json_f, 'r') as f:
        data=json.load(f)
    
    out_data = []
    for i in tqdm(range(len(data['data']))):
        dataset_dict = data['data'][i]
        dataset_dict['dataset'] = 'edina_test'

        out_data.append(dataset_dict)

    dump_file_summary = assign_info('test', out_data, 'edina_test', save_f)
    with open(output_json_f, 'w') as f:
        json.dump(dump_file_summary, f)