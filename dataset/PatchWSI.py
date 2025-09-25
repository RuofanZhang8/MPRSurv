"""
Class for bag-style dataloader
"""
from typing import Union
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.io import read_patch_data
from utils.func import sampling_data 
from .label_converter import MetaSurvData

class WSIPatchSurv(Dataset):
    r"""A patch dataset class for classification tasks (patient-level in general).

    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        table_path (string): The path of table with dataset labels, which has to be included. 
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    Return:
        index: The index of current item in the whole dataset.
        (feats, extra_data): Patch features and extra data.
        label: It contains typical survival labels, 'last follow-up time' and 'censorship status';
            censorship = 0 ---> uncersored, w/ event; censorship = 1 ---> cersored, w/o event.  
    """
    def __init__(self, patient_ids: list, patch_path: str, mode:str, meta_data:MetaSurvData,
        read_format:str='pt', ratio_sampling:Union[None,float,int]=None, **kws):
        super().__init__()
        if ratio_sampling is not None:
            print("[dataset] Patient-level sampling with ratio_sampling = {}".format(ratio_sampling))
            patient_ids, pid_left = sampling_data(patient_ids, ratio_sampling)
            print("[dataset] Sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))

        # assert mode in ['patch', 'cluster', 'graph']
        self.mode = mode
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        if self.mode == 'graph':
            assert 'graph_path' in kws
        self.kws = kws

        self.pids, self.pid2sids, self.pid2label = meta_data.collect_info_by_pids(patient_ids)

        self.meta_data = meta_data
        self.uid = self.pids
        self.read_path = patch_path
        self.read_format = read_format
        self.summary()

    def summary(self):
        print(f"[Dataset] WSIPatchSurv: in {self.mode} mode, avaiable patients count {self.__len__()}.")

    def get_meta_data(self):
        return self.meta_data

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid   = self.pids[index]
        sids  = self.pid2sids[pid]
        label = self.pid2label[pid]
        # get all data from one patient
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor(label).to(torch.float)

        if self.mode == 'patch':
            feats = []
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                if not osp.exists(full_path):
                    print(f"[WSIPatchSurv] warning: not found slide {sid}.")
                    continue
                feats.append(read_patch_data(full_path, dtype='torch'))

            feats = torch.cat(feats, dim=0).to(torch.float)
            return index, (feats, torch.Tensor([0])), label

        elif self.mode == 'slide':  ### read slide features
            patch_feats = []
            slide_feats = []
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                if not osp.exists(full_path):
                    print(f"[WSIPatchSurv] warning: not found slide {sid}.")
                    continue
                patch_feats.append(read_patch_data(full_path, dtype='torch'))  #,mode='slide'

                full_path_slide = osp.join(self.read_path.replace('features_conch_v15','slide_features_titan'), sid + '.' + self.read_format)
                if not osp.exists(full_path_slide):
                    print(f"[WSIPatchSurv] warning: not found slide feature {sid}.")
                    continue
                slide_feats.append(read_patch_data(full_path_slide, dtype='torch').unsqueeze(0))  #,mode='slide')

            patch_feats = torch.cat(patch_feats, dim=0).to(torch.float)

            slide_feats = torch.cat(slide_feats, dim=0).to(torch.float)

            return index, ([patch_feats,slide_feats], torch.Tensor([0])), label
        

        else:
            pass
            return None



