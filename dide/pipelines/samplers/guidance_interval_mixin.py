from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, cfg_strength, neg_cond, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        
        
    # def _inference_model_ss(self, model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, cfg_strength, neg_cond, cfg_interval, **kwargs):
    #     # if cfg_interval[0] <= t <= cfg_interval[1]:
    #     #     pred_v0, pred_v1, pred_v2, pred_v3 = super()._inference_model_ss(model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, **kwargs)

    #     #     neg_pred_0, neg_pred_1, neg_pred_2, neg_pred_3 = super()._inference_model_ss(model, x_t0, x_t1, x_t2, x_t3, t, neg_cond, neg_cond, neg_cond, **kwargs)

    #     #     return (1 + cfg_strength) * pred_v0 - cfg_strength * neg_pred_0, (1 + cfg_strength) * pred_v1 - cfg_strength * neg_pred_1, (1 + cfg_strength) * pred_v2 - cfg_strength * neg_pred_2, (1 + cfg_strength) * pred_v3 - cfg_strength * neg_pred_3
    #     # else:
    #     #     return super()._inference_model_ss(model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, **kwargs)
    #     pred_v0, pred_v1, pred_v2, pred_v3 = super()._inference_model_ss(model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, **kwargs)

    #     neg_pred_0, neg_pred_1, neg_pred_2, neg_pred_3 = super()._inference_model_ss(model, x_t0, x_t1, x_t2, x_t3, t, neg_cond, neg_cond, neg_cond, **kwargs)

    #     return (1 + cfg_strength) * pred_v0 - cfg_strength * neg_pred_0, (1 + cfg_strength) * pred_v1 - cfg_strength * neg_pred_1, (1 + cfg_strength) * pred_v2 - cfg_strength * neg_pred_2, (1 + cfg_strength) * pred_v3 - cfg_strength * neg_pred_3
    
        
    # def _inference_model_slat(self, model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, cfg_strength, neg_cond, cfg_interval, **kwargs):
        
    #     # if cfg_interval[0] <= t <= cfg_interval[1]:
        
    #     #     pred_v0, pred_v1, pred_v2, pred_v3 = super()._inference_model_slat(model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, **kwargs)

    #     #     neg_pred_0, neg_pred_1, neg_pred_2, neg_pred_3 = super()._inference_model_slat_nocond(model, x_t0, x_t1, x_t2, x_t3, t, neg_cond, cond2, cond3, **kwargs)

    #     #     return (1 + cfg_strength) * pred_v0 - cfg_strength * neg_pred_0, (1 + cfg_strength) * pred_v1 - cfg_strength * neg_pred_1, (1 + cfg_strength) * pred_v2 - cfg_strength * neg_pred_2, (1 + cfg_strength) * pred_v3 - cfg_strength * neg_pred_3
    #     # else:
            
    #     #     return super()._inference_model_slat(model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, **kwargs)
        
        
    #     pred_v0, pred_v1, pred_v2, pred_v3 = super()._inference_model_slat(model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, **kwargs)

    #     neg_pred_0, neg_pred_1, neg_pred_2, neg_pred_3 = super()._inference_model_slat_nocond(model, x_t0, x_t1, x_t2, x_t3, t, neg_cond, neg_cond, neg_cond, **kwargs)

    #     return (1 + cfg_strength) * pred_v0 - cfg_strength * neg_pred_0, (1 + cfg_strength) * pred_v1 - cfg_strength * neg_pred_1, (1 + cfg_strength) * pred_v2 - cfg_strength * neg_pred_2, (1 + cfg_strength) * pred_v3 - cfg_strength * neg_pred_3

        
    #     # return super()._inference_model_slat(model, x_t0, x_t1, x_t2, x_t3, t, cond1, cond2, cond3, **kwargs) 