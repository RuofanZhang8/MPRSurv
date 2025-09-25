import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import *

EPS = 1e-6
__all__ = [
    "logit_pooling", "FeatMIL", "VLFAN", "DeepMIL", "DSMIL", 
    "TransMIL", "ILRA", "DeepAttnMISL", "PatchGCN",'CPMIL'
]

def logit_pooling(logits, method):
    """
    logits: N x C logit for each patch
    method:
        - logit_topk: the top number of patches to use for pooling
        - logit_max: mean logits
        - logit_mean: max logits (act as logit_top1)
    """
    if method[:9] in ['logit_max', 'logit_top']:
        topk = 1 if method == 'logit_max' else int(method.split('top')[-1])
        # Sums logits across topk patches for each class, to get class prediction for each topk
        maxk = min(topk, logits.size(0)) # Ensures k is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
        values, _ = logits.topk(maxk, 0, True, True) # maxk x C
        pooled_logits = values.mean(dim=0, keepdim=True) # 1 x C logit scores
    elif method == 'logit_mean':
        pooled_logits = logits.mean(dim=0, keepdim=True) # 1 x C logit scores
    else:
        raise NotImplementedError(f"The pooling ({method}) is not implemented.")
    
    preds = pooled_logits.argmax(dim=1) # predicted class indices
    
    return preds, pooled_logits


class CSA(nn.Module):
    def __init__(self, dim_in=1024, dim_hid=256, use_feat_proj=True, drop_rate=0.25,pred_head='default', freeze_wsienc=True,**kwargs):
        super(CSA, self).__init__()
        if use_feat_proj:
            self.feat_proj = Feat_Projecter(dim_in, dim_in)
        else:
            self.feat_proj = None
    
        self.slide_level_encoder =None 
        self.cpnet = MPRSMIL(dim_in=dim_in, dim_hid=dim_hid, num_cls=1, dropout=drop_rate,use_feat_proj=False)
        self.revert_attention = RevertAttention()
        self.norevert_attention = NoRevertAttention()

        self.pred_head = pred_head
        
        self.final_head = nn.Linear(dim_in*2, dim_in)
        self.freeze_wsienc = freeze_wsienc
        self.alpha = 1.0

    def set_slide_level_encoder(self, slide_level_encoder):
        self.slide_level_encoder = slide_level_encoder
        if self.freeze_wsienc:
            print("[CPMIL] freeze the slide-level encoder.")
            for param in self.slide_level_encoder.parameters():
                param.requires_grad = False

    def forward(self, X, ret_with_attn=False,ret_with_feat=False):
        patch_feat, wsi_feat = X[0], X[1]

        wsi_feat = wsi_feat.mean(dim=1)  # [B, C]

        wsi_cp = self.cpnet(patch_feat)
        wsi_cp, wsi_feat = wsi_cp.unsqueeze(1), wsi_feat.unsqueeze(1)  # [1, C] -> [1, 1, C]
        cp_rt_output = self.revert_attention(wsi_cp, wsi_feat)
        wsi_feat = wsi_feat + cp_rt_output

        feat_rt_output = self.norevert_attention(wsi_cp, wsi_feat)
        wsi_cp = wsi_cp + feat_rt_output

        wsi_feat , wsi_cp = wsi_feat.squeeze(1), wsi_cp.squeeze(1)  # [1, C] -> [C]


        enhanced_output = self.final_head(torch.cat([wsi_feat,wsi_cp],dim=1))  # [B, C]
        if ret_with_feat :
            return enhanced_output, wsi_feat,wsi_cp
        else:
            return enhanced_output  # [B, 1, C] -> [B, C]
