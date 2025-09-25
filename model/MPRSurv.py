
import os.path as osp

import torch 
import torch.nn as nn
import torch.nn.functional as F

from model.prompt_learners import load_prompt_learner
from model.prompt_learners import load_prompt_adapter
from model.prompt_encoder import get_prompt_encoder
from model.utils_vl import load_vl_model_to_cpu
from model.utils_vl import Tokenizer
from model.deepmil import logit_pooling


class MPRSurv(nn.Module):
    def __init__(
        self,
        text_encoder_cfg,
        image_encoder_cfg,
        prompt_learner_cfg,
        pretrained_prompt_learner_cfg=None,
        info_prefix='MPRSurv-UNI',
        **kwargs,
    ) -> None:
        super().__init__()

        self.kwargs = kwargs
        print(f"[{info_prefix}] Found additional kwargs: {self.kwargs}.")
        assert 'MPRSurv_api' in kwargs, "Please specify `MPRSurv_api` in arguments."
        assert 'path_clip_model' in kwargs, "Please specify `path_clip_model` in arguments."

        print(text_encoder_cfg,image_encoder_cfg)
        vl_model = load_vl_model_to_cpu(
            text_encoder_cfg,
            image_encoder_cfg,
            root=kwargs['path_clip_model'],
            api=kwargs['MPRSurv_api'] 
        )
        
        self.text_tokenizer = Tokenizer(
            root=kwargs['path_clip_model'], 
            name=text_encoder_cfg['name'],
            api='TITAN',
        )

        # # titan tokenizer
        # self.text_tokenizer = vl_model.text_encoder.tokenizer
        
        # Language-end
        self.pmt_learner_name = prompt_learner_cfg['name']
        self.prompt_encoder = get_prompt_encoder(vl_model, api=kwargs['MPRSurv_api'])
        if self.pmt_learner_name == 'CoOp':
            self.prompt_learner, pretrained_text_features = self._build_prompt_learner(
                prompt_learner_cfg, pretrained_prompt_learner_cfg
            )
            if pretrained_text_features is not None:
                pretrained_text_features = pretrained_text_features.detach().clone()
                self.register_buffer("pretrained_text_features", pretrained_text_features, persistent=False)
                print("[MPRSurv] warning: skip CoOp-based prompt learner and use pretrained text features.")
        
        else:
            raise ValueError(f"{self.pmt_learner_name} is not a valid name of prompt learner.")

        # Vision-end

        if hasattr(vl_model, 'vision_encoder'):
            assert kwargs['MPRSurv_api'] == 'TITAN'
            self.mil_encoder = vl_model.vision_encoder
        else:
            raise ValueError(f"[{info_prefix}] `vision_model`, `visual`, or `vision_encoder` is not found in {vl_model}.")
        
        self.text_encoder_cfg = text_encoder_cfg
        self.image_encoder_cfg = image_encoder_cfg
        self.prompt_learner_cfg = prompt_learner_cfg

        self.logit_scale = torch.tensor(4.0315)   #vl_model.logit_scale  #

    def _build_prompt_learner(self, prompt_learner_cfg, pretrained_prompt_learner_cfg):
        _prompt_learner_cfg = prompt_learner_cfg.copy()
        _prompt_learner_cfg.update(dict(
            tokenizer = self.text_tokenizer, 
            text_config = self.prompt_encoder.text_config,
            token_embedding = self.prompt_encoder.token_embedding
        ))
        prompt_learner = load_prompt_learner(_prompt_learner_cfg['method'], _prompt_learner_cfg)

        # if use pretrained text prompts
        pretrained_text_features = None
        if _prompt_learner_cfg['pretrained']:
            assert pretrained_prompt_learner_cfg is not None, "Please specify `config` for `pretrained_prompt_learner`."
            prompt_learner.load_pretrained_parameters(pretrained_prompt_learner_cfg['ckpt'])
            
            # if there is no trainable parameter, pre-compute the fixed text features
            if _prompt_learner_cfg['frozen_context_embeds'] and _prompt_learner_cfg['frozen_rank_embeds']:
                with torch.no_grad():
                    pretrained_text_features = self.compute_text_features_with_coop(prompt_learner)

        return prompt_learner, pretrained_text_features

    def _build_prompt_adapter(self, prompt_learner_cfg, pretrained_prompt_learner_cfg):
        _prompt_learner_cfg = prompt_learner_cfg.copy()
        _pretrained_prompt_learner_cfg = pretrained_prompt_learner_cfg.copy()

        # if use CoOp-pretrained text prompts for Adapter
        pretrained_text_features = None
        if _prompt_learner_cfg['pretrained']:
            _pretrained_prompt_learner_cfg['pretrained'] = True
            _, pretrained_text_features = self._build_prompt_learner(
                _pretrained_prompt_learner_cfg, {'ckpt': _pretrained_prompt_learner_cfg['ckpt']}
            )
            assert pretrained_text_features is not None, "Found empty `pretrained_text_features`."
            pretrained_text_features = pretrained_text_features.detach().clone()

        _prompt_learner_cfg.update(dict(
            tokenizer = self.text_tokenizer,
            num_prompts = _prompt_learner_cfg['num_ranks'],
            pretrained_prompt_features = pretrained_text_features,
        ))
        prompt_adapter = load_prompt_adapter(self.prompt_encoder, _prompt_learner_cfg)

        return prompt_adapter

    def compute_text_features_with_coop(self, prompt_learner):
        sentence_embeds = prompt_learner()
        pseudo_sentence_tokens = prompt_learner.pseudo_sentence_tokens
        text_features = self.prompt_encoder(
            prompts_embedding=sentence_embeds, 
            prompts_pseudo_tokens=pseudo_sentence_tokens
        )
        return text_features

    def forward_text_only(self):
        # use pretrained_text_features if exists
        if hasattr(self, 'pretrained_text_features'):
            return self.pretrained_text_features.clone()

        if self.pmt_learner_name == 'CoOp':
            text_features = self.compute_text_features_with_coop(self.prompt_learner)
        else:
            text_features = None
            pass

        return text_features

    def encode_instances(self, X,ret_with_feat=False):
        if ret_with_feat:
            return self.mil_encoder(X,ret_with_feat=True)
        else:
            return self.mil_encoder(X)

    def get_logit_scale(self):
        return self.logit_scale.exp()

    def forward(self, X,ret_with_feat=False):
        """
        X: a bag with instance feature vectors, with shape of [1, N, feat_dim].
        """
        text_features = self.forward_text_only()
        text_features = F.normalize(text_features, dim=-1) # [num_ranks, emb_dim]
        if ret_with_feat:
            image_features,wsi_feat,wsi_cp = self.encode_instances(X,ret_with_feat=True)
        else:
            image_features = self.encode_instances(X,ret_with_feat)
        image_features = F.normalize(image_features, dim=-1) # [1, emb_dim] or [N, emb_dim]

        logits = image_features @ text_features.t()

        # at zero-shot mode, mil_encoder is Identity and logits come from all instances
        if logits.shape[0] > 1:
            _, logits = logit_pooling(logits, self.image_encoder_cfg['pooling'])

        if ret_with_feat:
            return logits, wsi_feat, wsi_cp
        else:
            return logits, image_features, text_features
