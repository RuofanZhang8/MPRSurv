#####################################
# Evaluator for survival models
#####################################
import numpy as np
import torch
import torch.nn.functional as F

from loss.utils import load_surv_loss_func
from dataset.label_converter import MetaSurvData
from .cindex import concordance_index


AVAILABLE_LOSSES_LIST = {
    'loss_rank': load_surv_loss_func('rank_loss'),
    'loss_recon': load_surv_loss_func('recon_loss'),
    'loss_recon_org': load_surv_loss_func('recon_loss', alpha=0.0),
    'loss_mle': load_surv_loss_func('SurvMLE'),
    'loss_mle_org': load_surv_loss_func('SurvMLE', alpha=0.0),
    'loss_if_mle': load_surv_loss_func('SurvIFMLE'),
    'loss_if_mle_org': load_surv_loss_func('SurvIFMLE', alpha=0.0),
    'loss_ple': load_surv_loss_func('SurvPLE'),
}


def load_SurvivalEVAL(meta_data: MetaSurvData, time_coordinates=None, predict_time_method='Mean'):
    assert predict_time_method in ['Mean', 'Median']

    if time_coordinates is None:
        time_coordinates = meta_data.time_coordinates
    data_train = meta_data.get_patient_data(split='train', ret_columns=['t', 'e'])
    data_test = meta_data.get_patient_data(split='test', ret_columns=['t', 'e'])
    survival_outputs = np.ones((1, len(time_coordinates)), dtype=np.float32) # temp survival preds
    
    from eval.SurvivalEVAL import SurvivalEvaluator
    evaler = SurvivalEvaluator(
        survival_outputs, time_coordinates, 
        data_test.t.values, data_test.e.values,
        data_train.t.values, data_train.e.values,
        predict_time_method=predict_time_method,
    )
    return evaler


class NLLSurv_Evaluator(object):
    """
    NLLSurv_Evaluator for NLL (Negative Log-Likelihood)-based models, or discrete survival models.
    """
    def __init__(self, prediction_type:str, backend='SurvivalEVAL', **kws):
        super().__init__()
        self.type = prediction_type
        self.kws = kws
        self.backend = backend
        assert self.type in ['hazard', 'incidence'], "The `prediction_type` should be hazard or incidence."

        self.aux_evaluator = None
        self.meta_data = None
        if self.backend == 'SurvivalEVAL':
            assert 'meta_data' in self.kws, "Please specify `meta_data` when using SurvivalEVAL as backend."
            self.meta_data = self.kws['meta_data']
            self.aux_evaluator = load_SurvivalEVAL(self.meta_data, predict_time_method='Mean')

            self.valid_functions = {
                'c_index': self._aux_c_index,
                'c_index2': self._c_index,
                'loss': self._loss_mle_org,
                'loss_mle': self._loss_mle,
                'loss_mle_org': self._loss_mle_org,
                'IBS': self._aux_integrated_brier_score,
                'MAE': self._aux_mae,
                'D_calibration': self._aux_distribution_calibration,
            }
            self.valid_metrics = ['c_index', 'loss', 'loss_mle', 'loss_mle_org', 'IBS', 'MAE', 'D_calibration', 'c_index2']
        else:
            self.valid_functions = {
                'c_index': self._c_index,
                'loss': self._loss_mle_org,
                'loss_mle': self._loss_mle,
                'loss_mle_org': self._loss_mle_org,
            }
            self.valid_metrics = ['c_index', 'loss', 'loss_mle', 'loss_mle_org']

        print(f"[NLLSurv Evaluator] use backend = {self.backend} for evaluation.")
        print(f"[NLLSurv Evaluator] got additional kws: {self.kws}.")
        print(f"[NLLSurv Evaluator] This evaluator is designed for {self.type} prediction models.")

    def _check_metrics(self, metrics):
        for m in metrics:
            assert m in self.valid_metrics, f"[NLLSurv Evaluator] got an invalid metric name: {m}."

    def _pre_compute(self, data):
        self.y = data['y']
        self.t = data['y'][:, 0]
        self.e = data['y'][:, 1]
        self.c = 1.0 - data['y'][:, 1]
        # only used for computing CI
        if 'avg_y_hat' in data:
            self.y_hat = data['avg_y_hat']
        else:
            self.y_hat = data['y_hat']

        cur_uid = data['uid']

        # get raw prediction
        if 'raw_y_hat' in data:
            self.raw_y_hat = data['raw_y_hat']
        else:
            self.raw_y_hat = None

        if self.type == 'incidence':
            pred_CIF = torch.cumsum(self.y_hat, dim=1)
            self.survival_hat = 1.0 - pred_CIF
            self.survival_hat[self.survival_hat < 0] = 0
        elif self.type == 'hazard':
            self.survival_hat = torch.cumprod(1.0 - self.y_hat, dim=1)
            self.survival_hat[self.survival_hat < 0] = 0
        else:
            self.survival_hat = None

        if self.backend == 'SurvivalEVAL':
            # reset the input (pred) of aux_evaluator for evaluation
            self.aux_evaluator.predicted_curves = self.survival_hat

            # reset the input (true) of aux_evaluator for evaluation
            actual_label = self.meta_data.get_patient_data(pids=cur_uid, ret_columns=['t', 'e'])
            assert len(actual_label) == len(self.survival_hat), "Pred and Label do not match in dimension."
            self.aux_evaluator.actual_survival_time = actual_label.t.values
            self.aux_evaluator.actual_survival_event = actual_label.e.values

    def _c_index(self):
        y_true = self.y.numpy()
        y_pred = self.y_hat.numpy()
        return concordance_index(y_true, y_pred, type_pred=self.type)

    def _loss_mle(self):
        loss_to_select = None
        if self.type == 'incidence':
            loss_to_select = 'loss_if_mle'
        elif self.type == 'hazard':
            loss_to_select = 'loss_mle'

        _mle_loss = AVAILABLE_LOSSES_LIST[loss_to_select]
        loss = _mle_loss(self.y_hat, self.t, self.e)
        return loss.item()

    def _loss_mle_org(self):
        loss_to_select = None
        if self.type == 'incidence':
            loss_to_select = 'loss_if_mle_org'
        elif self.type == 'hazard':
            loss_to_select = 'loss_mle_org'

        _mle_loss = AVAILABLE_LOSSES_LIST[loss_to_select]
        loss = _mle_loss(self.y_hat, self.t, self.e)
        return loss.item()

    # the following functions starting with `_aux` is for `self.aux_evaluator`.
    # When backend = 'SurvivalEVAL', `self.aux_evaluator` is from `eval.SurvivalEVAL.SurvivalEvaluator`.
    def _aux_c_index(self, ties='All'):
        if self.backend == 'SurvivalEVAL':
            cindex, concordant_pairs, total_pairs = self.aux_evaluator.concordance(ties=ties)
        else:
            raise NotImplementedError(f"C-Index is not implemented for backend {self.backend}.")
        return cindex

    def _aux_integrated_brier_score(self, IPCW_weighted=True):
        if self.backend == 'SurvivalEVAL':
            ibs = self.aux_evaluator.integrated_brier_score(
                num_points=None, IPCW_weighted=IPCW_weighted, draw_figure=False
            )
        else:
            raise NotImplementedError(f"Integrated Brier Score is not implemented for backend {self.backend}.")
        return ibs

    def _aux_mae(self, method='Hinge', reduction=True):
        if self.backend == 'SurvivalEVAL':
            mae_score = self.aux_evaluator.mae(method=method, reduction=reduction)
        else:
            raise NotImplementedError(f"MAE-Hinge is not implemented for backend {self.backend}.")
        return mae_score

    def _aux_distribution_calibration(self):
        # p_value >= 0.05 means distribution-calibrated
        # p_value <  0.05 means NOT distribution-calibrated
        if self.backend == 'SurvivalEVAL':
            p_value, bin_statistics = self.aux_evaluator.d_calibration()
        else:
            raise NotImplementedError(f"D-Calibration is not implemented for backend {self.backend}.")
        return p_value

    def _aux_predicted_event_times(self):
        if self.backend == 'SurvivalEVAL':
            predicted_event_times = self.aux_evaluator.predicted_event_times
        else:
            raise NotImplementedError(f"`predicted_event_times` is not implemented for backend {self.backend}.")
        return predicted_event_times

    def _eval_ext_loss(self, loss_name, loss_func, **kws):
        t, e = self.t.unsqueeze(-1), self.e.unsqueeze(-1)
        weight = kws['weight'] if 'weight' in kws else 1
        if loss_name == 'SurvEMD':
            cur_logit_scale = kws['logit_scale']
            loss = weight * loss_func(self.y_hat, t, e, cur_logit_scale)
        elif loss_name == 'SurvT2I':
            cur_logit_scale = kws['logit_scale']
            loss = weight * loss_func(self.raw_y_hat, t, e, cur_logit_scale)
        elif loss_name == 'QueryDiv':
            loss = weight * loss_func()
        elif loss_name == 'NLLloss':
            loss =  loss_func(self.raw_y_hat, t, e)['loss']
        else:
            loss = weight * loss_func(self.y_hat, t, e)

        return loss.item()

    def compute(self, data, metrics, kws_ext_loss=None, **kws):
        self._check_metrics(metrics)
        self._pre_compute(data)
        res_metrics = dict()
        for m in metrics:
            res_metrics[m] = self.valid_functions[m]()
        
        if kws_ext_loss is not None:
            assert isinstance(kws_ext_loss, dict)
            for loss_name, loss_func in kws_ext_loss.items():
                weight = kws['loss_weight'][loss_name] if 'loss_weight' in kws else 1
                logit_scale = kws['logit_scale'] if 'logit_scale' in kws else 10.0
                # evaluation on CPU
                if isinstance(logit_scale, torch.Tensor):
                    logit_scale = logit_scale.cpu()
                res_metrics['loss_'+loss_name] = self._eval_ext_loss(
                    loss_name, loss_func, 
                    weight=weight, 
                    logit_scale=logit_scale
                )
        
        return res_metrics

