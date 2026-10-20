from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
import random

class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    
    def _inference_ss(self, model, x_t0, t, cond1, **kwargs):
        t = torch.tensor([1000 * t] * x_t0.shape[0], device=x_t0.device, dtype=torch.float32)
        return model.forward_ss(x_t0, t, cond1)


    def _inference_ss_nocond(self, model, x_t0, t, cond1, **kwargs):
        t = torch.tensor([1000 * t] * x_t0.shape[0], device=x_t0.device, dtype=torch.float32)
        return model.forward_ss_nocond(x_t0, t, cond1)


    def _inference_model_ss(self, model, x_t0, t, cond1, cfg_strength, neg_cond, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred_v0 = self._inference_ss(model, x_t0, t, cond1, **kwargs)
            # neg_pred_0 = self._inference_model(model, x_t0, t, neg_cond, **kwargs)
            # neg_pred_1 = self._inference_model(model, x_t1, t, neg_cond, **kwargs)
            # neg_pred_2 = self._inference_model(model, x_t2, t, neg_cond, **kwargs)
            # neg_pred_3 = self._inference_model(model, x_t3, t, neg_cond, **kwargs)
            
            neg_pred_0 = self._inference_ss_nocond(model, x_t0, t, neg_cond, **kwargs)

            return (1 + cfg_strength) * pred_v0 - cfg_strength * neg_pred_0
        else:
            
            return self._inference_ss(model, x_t0, t, cond1, **kwargs)


    def _inference_slat(self, model, x_t0, x_t1, x_t2, x_t3, x_t4, t, cond_cnt, cond_txu, cond_col, **kwargs):
        t = torch.tensor([1000 * t] * x_t1.shape[0], device=x_t1.device, dtype=torch.float32)
        return model.forward_slat(x_t0, x_t1, x_t2, x_t3, x_t4, t, cond_cnt, cond_txu, cond_col)


    def _inference_slat_nocond(self, model, x_t0, x_t1, x_t2, x_t3, x_t4, t, cond_cnt, cond_txu, cond_col, **kwargs):
        t = torch.tensor([1000 * t] * x_t1.shape[0], device=x_t1.device, dtype=torch.float32)
        return model.forward_slat_nocond(x_t0, x_t1, x_t2, x_t3, x_t4, t, cond_cnt, cond_txu, cond_col)


    def _inference_model_slat(self, model, x_t0, x_t1, x_t2, x_t3, x_t4, t, cond_cnt, cond_txu, cond_col, cfg_strength, neg_cond, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred_v0, pred_v1, pred_v2, pred_v3, pred_v4 = self._inference_slat(model, x_t0, x_t1, x_t2, x_t3, x_t4, t, cond_cnt, cond_txu, cond_col, **kwargs)
            # neg_pred_0 = self._inference_model(model, x_t0, t, neg_cond, **kwargs)
            # neg_pred_1 = self._inference_model(model, x_t1, t, neg_cond, **kwargs)
            # neg_pred_2 = self._inference_model(model, x_t2, t, neg_cond, **kwargs)
            # neg_pred_3 = self._inference_model(model, x_t3, t, neg_cond, **kwargs)
            neg_pred_0, neg_pred_1, neg_pred_2, neg_pred_3, neg_pred_4 = self._inference_slat_nocond(model, x_t0, x_t1, x_t2, x_t3, x_t4, t, neg_cond, neg_cond, neg_cond, **kwargs)

            return (1 + cfg_strength) * pred_v0 - cfg_strength * neg_pred_0, (1 + cfg_strength) * pred_v1 - cfg_strength * neg_pred_1, (1 + cfg_strength) * pred_v2 - cfg_strength * neg_pred_2, (1 + cfg_strength) * pred_v3 - cfg_strength * neg_pred_3, (1 + cfg_strength) * pred_v4 - cfg_strength * neg_pred_4
        else:
            return self._inference_slat(model, x_t0, x_t1, x_t2, x_t3, x_t4, t, cond_cnt, cond_txu, cond_col, **kwargs)

    def _get_model_prediction_ss(self, model, x_t0, t, cond, cfg_strength, **kwargs):
        pred_v0 = self._inference_model_ss(model, x_t0, t, cond, cfg_strength, **kwargs)
        pred_x_0, pred_eps_0 = self._v_to_xstart_eps(x_t=x_t0, t=t, v=pred_v0)

        return pred_x_0, pred_eps_0, pred_v0
    


    def _get_model_prediction_slat(self, model, x_t0, x_t1, x_t2, x_t3, x_t4, t, cond_cnt, cond_txu, cond_col, cfg_strength, **kwargs):
        pred_v0, pred_v1, pred_v2, pred_v3, pred_v4 = self._inference_model_slat(model, x_t0, x_t1, x_t2, x_t3, x_t4, t, cond_cnt, cond_txu, cond_col, cfg_strength, **kwargs)
        pred_x_0, pred_eps_0 = self._v_to_xstart_eps(x_t=x_t0, t=t, v=pred_v0)
        pred_x_1, pred_eps_1 = self._v_to_xstart_eps(x_t=x_t1, t=t, v=pred_v1)
        pred_x_2, pred_eps_2 = self._v_to_xstart_eps(x_t=x_t2, t=t, v=pred_v2)
        pred_x_3, pred_eps_3 = self._v_to_xstart_eps(x_t=x_t3, t=t, v=pred_v3)
        pred_x_4, pred_eps_4 = self._v_to_xstart_eps(x_t=x_t4, t=t, v=pred_v4)
        return pred_x_0, pred_eps_0, pred_v0, pred_x_1, pred_eps_1, pred_v1, pred_x_2, pred_eps_2, pred_v2, pred_x_3, pred_eps_3, pred_v3, pred_x_4, pred_eps_4, pred_v4
    


    @torch.no_grad()
    def sample_once_ss(
        self,
        model,
        x_t0,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        cfg_strength = 3.5,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps_0, pred_v_0 = self._get_model_prediction_ss(model, x_t0, t, cond, cfg_strength, **kwargs)
        pred_x_prev_0 = x_t0 - (t - t_prev) * pred_v_0

        return edict({"pred_x_prev": pred_x_prev_0, "pred_x_0": pred_x_0})
    

    @torch.no_grad()
    def sample_once_slat(
        self,
        model,
        x_t0,
        x_t1,
        x_t2,
        x_t3,
        x_t4,
        t: float,
        t_prev: float,
        cond_cnt: Optional[Any] = None,
        cond_txu: Optional[Any] = None,
        cond_col: Optional[Any] = None,
        cfg_strength = 3.5,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps_0, pred_v_0, pred_x_1, pred_eps_1, pred_v_1, pred_x_2, pred_eps_2, pred_v_2, pred_x_3, pred_eps_3, pred_v_3, pred_x_4, pred_eps_4, pred_v_4  = self._get_model_prediction_slat(model, x_t0, x_t1, x_t2, x_t3, x_t4, t, cond_cnt, cond_txu, cond_col, cfg_strength, **kwargs)
        pred_x_prev_0 = x_t0 - (t - t_prev) * pred_v_0
        pred_x_prev_1 = x_t1 - (t - t_prev) * pred_v_1
        pred_x_prev_2 = x_t2 - (t - t_prev) * pred_v_2
        pred_x_prev_3 = x_t3 - (t - t_prev) * pred_v_3 
        pred_x_prev_4 = x_t4 - (t - t_prev) * pred_v_4 
        return edict({"pred_x_prev": pred_x_prev_0, "pred_x_0": pred_x_0}), edict({"pred_x_prev": pred_x_prev_1, "pred_x_0": pred_x_1}), edict({"pred_x_prev": pred_x_prev_2, "pred_x_0": pred_x_2}), edict({"pred_x_prev": pred_x_prev_3, "pred_x_0": pred_x_3}), edict({"pred_x_prev": pred_x_prev_4, "pred_x_0": pred_x_4})
    
    
    @torch.no_grad()
    def sample_style(
        self,
        model,
        noise1,
        cond_cnt: dict,
        cond_txu: dict,
        cond_col: dict,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        cfg_strength=3.5,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample0, sample1, sample2, sample3, sample4 = noise1, noise1, noise1, noise1, noise1
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        start = 0

        for t, t_prev in tqdm(t_pairs[:], desc="Sampling", disable=not verbose):

            k1, k2 = start % len(cond_cnt), start % len(cond_txu)
            out0, out1, out2, out3, out4 = self.sample_once_slat(model, sample0, sample1, sample2, sample3, sample4, t, t_prev, cond_cnt[k1:k1+1], cond_txu[k2:k2+1], cond_col[k2:k2+1], cfg_strength = cfg_strength, **kwargs)
            sample0, sample1, sample2, sample3, sample4 = out0.pred_x_prev, out1.pred_x_prev, out2.pred_x_prev, out3.pred_x_prev, out4.pred_x_prev
            start = start + 1
            ret.pred_x_t.append(out0.pred_x_prev)
            ret.pred_x_0.append(out0.pred_x_0)
    
        ret.samples = sample0
        return ret
    
    
    @torch.no_grad()
    def sample_structure(
        self,
        model,
        noise,
        cond: dict,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        cfg_strength=9.5,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        start = 0
        
        for t, t_prev in tqdm(t_pairs[:], desc="Sampling", disable=not verbose):

            k1 = start % len(cond)
            out = self.sample_once_ss(model, sample, t, t_prev, cond[k1:k1+1], cfg_strength = cfg_strength, **kwargs)
            sample = out.pred_x_prev
            start = start + 1
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
                
            
        ret.samples = sample
        
        return ret




class FlowEulerGuidanceIntervalSampler(FlowEulerSampler):

    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
    

    @torch.no_grad()
    def sample_style(
        self,
        model,
        noise1,
        condcnt,
        condtxu,
        condcol,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        neg_cond = condcnt['neg_cond']
        condcnt = condcnt['cond']
        condtxu = condtxu['cond']
        condcol = condcol['cond']
   
        return super().sample_style(model, noise1, condcnt, condtxu, condcol, steps, rescale_t, verbose, cfg_strength=cfg_strength, neg_cond=neg_cond, cfg_interval=cfg_interval, **kwargs)
    
    @torch.no_grad()
    def sample_structure(
        self,
        model,
        noise,
        cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        neg_cond = cond['neg_cond']
        cond = cond['cond']
    

        return super().sample_structure(model, noise, cond, steps, rescale_t, verbose, cfg_strength=cfg_strength, neg_cond=neg_cond, cfg_interval=cfg_interval, **kwargs)
    

