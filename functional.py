
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from plotting import *
from task_configs import TASK_MAP
from transfers import functional_transfers, finetuned_transfers

from functools import partial

import IPython


def get_functional_loss(config="F_gt_mse", mode='standard', **kwargs):
    if isinstance(mode, str):
        mode = {
            "standard": FunctionalLoss, 
            "mixing": MixingFunctionalLoss, 
            "normalized": NormalizedFunctionalLoss,
            "curriculum": CurriculumFunctionalLoss,
        }[mode]
    return mode(config=config, **kwargs)


### FUNCTIONAL LOSS CONFIGS
(f, F, g, G, s, S, CE, EC, DE, ED, h, H, n, npstep, RC, k, a, r, d, KC, k3C, Ck3, nr, rn, k3N, Nk3, Er) = functional_transfers
# (f, F, g, G, s, S, CE, EC, DE, ED, h, H, n, RC, k, a, r, d, KC, k3C, Ck3, nr, rn, k3N, Nk3, Er,) = finetuned_transfers

loss_configs = {
    "vid_F_EC_a_x": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC(a(x))": lambda y, y_hat, x: EC(a(x)), 
                "F(EC(a(x)))": lambda y, y_hat, x: F(EC(a(x))), 
            }
        ),
    "vid_F_RC_x": 
        (
            {
                "y -> F(RC(x))": lambda y, y_hat, x, norm: norm(y, F(RC(x))),
            },
            {
                "RC(x)": lambda y, y_hat, x: RC(x), 
                "F(RC(x))": lambda y, y_hat, x: F(RC(x)), 
            }
        ),
    "vid_F_f_S_a_x": 
        (
            {
                "y -> F(f(S(a(x)))": lambda y, y_hat, x, norm: norm(y, F(f(S(a(x))))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "s(a(x)": lambda y, y_hat, x: S(a(x)), 
                "f(s(a(x))": lambda y, y_hat, x: f(S(a(x))), 
                "F(f(S(a(x)))": lambda y, y_hat, x: F(f(S(a(x)))), 
            }
        ),
    "vid_all_three": 
        (
            {
                "y -> F(f(S(a(x)))": lambda y, y_hat, x, norm: norm(y, F(f(S(a(x))))),
                "y -> F(RC(x))": lambda y, y_hat, x, norm: norm(y, F(RC(x))),
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC(a(x))": lambda y, y_hat, x: EC(a(x)), 
                "F(EC(a(x)))": lambda y, y_hat, x: F(EC(a(x))), 
                "S(a(x)": lambda y, y_hat, x: S(a(x)), 
                "f(S(a(x))": lambda y, y_hat, x: f(S(a(x))), 
                "F(f(S(a(x)))": lambda y, y_hat, x: F(f(S(a(x)))), 
                "F(RC(x))": lambda y, y_hat, x: F(RC(x)), 
                "RC(x)": lambda y, y_hat, x: RC(x), 
            }
        ),
    "vid_C_two": 
        (
            {
                "y -> F(RC(x))": lambda y, y_hat, x, norm: norm(y, F(RC(x))),
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC(a(x))": lambda y, y_hat, x: EC(a(x)), 
                "F(EC(a(x)))": lambda y, y_hat, x: F(EC(a(x))), 
                "F(RC(x))": lambda y, y_hat, x: F(RC(x)),
                "RC(x)": lambda y, y_hat, x: RC(x), 

            }
        ),
    "vid_F_KC_k_x": 
        (
            {
                "y -> F(KC(k(x)))": lambda y, y_hat, x, norm: norm(y, F(KC(k(x)))),
            },
            {
                "F(KC(k(x)))": lambda y, y_hat, x: F(KC(k(x))), 
                "KC(k(x))": lambda y, y_hat, x: KC(k(x)), 
                "k(x)": lambda y, y_hat, x: k(x), 
            }
        ),
    "gt_mse": 
        (
            {
                "f(y) -> f(y^)": lambda y, y_hat, x, norm: norm(f(y), f(y_hat)),
            },
            {
                "f(y)": lambda y, y_hat, x: f(y), 
                "f(y^)": lambda y, y_hat, x: f(y_hat), 
            }
        ),
    "wGTinflux_A_B_C_normalized_delay": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "f(y) -> EC(a(x))": lambda y, y_hat, x, norm: norm(f(y), EC(a(x))),
                "CE[f(y)] -> a(x)": lambda y, y_hat, x, norm: norm(CE(f(y)), a(x)),
            },
            {
                "f(y)": lambda y, y_hat, x: f(y), 
                "CE(f(y))": lambda y, y_hat, x: CE(f(y)), 
                "a(x)": lambda y, y_hat, x: a(x),
                "EC(a(x))": lambda y, y_hat, x: EC(a(x)),
                "F(EC(a(x)))": lambda y, y_hat, x: F(EC(a(x))),
                "F(f(y))": lambda y, y_hat, x: F(f(y)),
            }
        ),
    "wGTinflux_A_B_normalized_delay": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "f(y) -> EC(a(x))": lambda y, y_hat, x, norm: norm(f(y), EC(a(x))),
            },
            {
                "f(y)": lambda y, y_hat, x: f(y), 
                "a(x)": lambda y, y_hat, x: a(x),
                "EC(a(x))": lambda y, y_hat, x: EC(a(x)),
                "F(EC(a(x)))": lambda y, y_hat, x: F(EC(a(x))),
                "F(f(y))": lambda y, y_hat, x: F(f(y)),
            }
        ),
    "wGTinflux_A_C_normalized_delay": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "f(y) -> EC(a(x))": lambda y, y_hat, x, norm: norm(f(y), EC(a(x))),
            },
            {
                "f(y)": lambda y, y_hat, x: f(y), 
                "a(x)": lambda y, y_hat, x: a(x),
                "EC(a(x))": lambda y, y_hat, x: EC(a(x)),
                "F(EC(a(x)))": lambda y, y_hat, x: F(EC(a(x))),
                "F(f(y))": lambda y, y_hat, x: F(f(y)),
            }
        ),
    "wGTinflux_A_percepcurv": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "f(y) -> f(F(EC(a(x))))": lambda y, y_hat, x, norm: norm(f(y), f(F(EC(a(x))))),
            },
            {
                "f(y)": lambda y, y_hat, x: f(y), 
                "a(x)": lambda y, y_hat, x: a(x),
                "EC(a(x))": lambda y, y_hat, x: EC(a(x)),
                "F(EC(a(x)))": lambda y, y_hat, x: F(EC(a(x))),
                "f(F(EC(a(x))))": lambda y, y_hat, x: f(F(EC(a(x)))),
            }
        ),
    "wGTinflux_A_percepcurv": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "f(y) -> f(F(EC(a(x))))": lambda y, y_hat, x, norm: norm(f(y), f(F(EC(a(x))))),
            },
            {
                "f(y)": lambda y, y_hat, x: f(y), 
                "a(x)": lambda y, y_hat, x: a(x),
                "EC(a(x))": lambda y, y_hat, x: EC(a(x)),
                "F(EC(a(x)))": lambda y, y_hat, x: F(EC(a(x))),
                "f(F(EC(a(x))))": lambda y, y_hat, x: f(F(EC(a(x)))),
            }
        ),
    "delayed_GT_curvpercep_cycle_split": 
        (
            {
                "f(y) -> f(y^)": lambda y, y_hat, x, norm: norm(f(y), f(y_hat)),
                "F(f(y)) -> y_frozen": lambda y, y_hat, x, norm: norm(F(f(y)), y.detach()),
                "F(f(y))_frozen -> y": lambda y, y_hat, x, norm: norm(F(f(y)).detach(), y),
            },
            {
                "f(y)": lambda y, y_hat, x: f(y), 
                "f(y^)": lambda y, y_hat, x: f(y_hat), 
                "F(f(y))": lambda y, y_hat, x: F(f(y)), 
            }
        ),
    "wGTinflux_curvA_depthA": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "y -> G(ED(a(x)))": lambda y, y_hat, x, norm: norm(y, G(ED(a(x)))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)),
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))),
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)),
                "G(ED[a(x)])": lambda y, y_hat, x: G(ED(a(x))),
            }
        ),
    "wGTinflux_curvA_depthviacurvA": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "y -> G(h(EC(a(x))))": lambda y, y_hat, x, norm: norm(y, G(h(EC(a(x))))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)),
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))),
                "h(EC[a(x)])": lambda y, y_hat, x: h(EC(a(x))),
                "G(h(EC[a(x)]))": lambda y, y_hat, x: G(h(EC(a(x)))),
            }
        ),
    "wGTinflux_curvAB_depthAB": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "y -> G(ED(a(x)))": lambda y, y_hat, x, norm: norm(y, G(ED(a(x)))),
                "f(y) -> EC(a(x))": lambda y, y_hat, x, norm: norm(f(y), EC(a(x))),
                "g(y) -> ED(a(x))": lambda y, y_hat, x, norm: norm(g(y), ED(a(x))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)),
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))),
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)),
                "G(ED[a(x)])": lambda y, y_hat, x: G(ED(a(x))),
                "f(y)": lambda y, y_hat, x: f(y),
                "g(y)": lambda y, y_hat, x: g(y),
            }
        ),
    "wGTinflux_curvA_depthA_trianglecurv2depth": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "y -> G(ED(a(x)))": lambda y, y_hat, x, norm: norm(y, G(ED(a(x)))),
                "h(f(y)) -> g(y)": lambda y, y_hat, x, norm: norm(h(f(y)), g(y)),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)),
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))),
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)),
                "G(ED[a(x)])": lambda y, y_hat, x: G(ED(a(x))),
                "f(y)": lambda y, y_hat, x: f(y),
                "h(f(y))": lambda y, y_hat, x: h(f(y)),
                "g(y)": lambda y, y_hat, x: g(y),
            }
        ),
    "wGTinflux_curvA_depthA_trianglecurv2depth_gt": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "y -> G(ED(a(x)))": lambda y, y_hat, x, norm: norm(y, G(ED(a(x)))),
                "h(f(y)) -> ED(a(x))": lambda y, y_hat, x, norm: norm(h(f(y)), ED(a(x))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)),
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))),
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)),
                "G(ED[a(x)])": lambda y, y_hat, x: G(ED(a(x))),
                "f(y)": lambda y, y_hat, x: f(y),
                "h(f(y))": lambda y, y_hat, x: h(f(y)),
            }
        ),
    "wGTinflux_curvA_depthA_triangledepth2curv_gt": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "y -> G(ED(a(x)))": lambda y, y_hat, x, norm: norm(y, G(ED(a(x)))),
                "H(g(y)) -> EC(a(x))": lambda y, y_hat, x, norm: norm(H(g(y)), EC(a(x))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)),
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))),
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)),
                "G(ED[a(x)])": lambda y, y_hat, x: G(ED(a(x))),
                "g(y)": lambda y, y_hat, x: g(y),
                "H(g(y))": lambda y, y_hat, x: H(g(y)),
            }
        ),
    "wGTinflux_A_trianglecurv2depth": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "h(f(y)) -> g(y)": lambda y, y_hat, x, norm: norm(h(f(y)), g(y)),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)),
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))),
                "f(y)": lambda y, y_hat, x: f(y),
                "h(f(y))": lambda y, y_hat, x: h(f(y)),
                "g(y)": lambda y, y_hat, x: g(y),
            }
        ),
    "wGTinflux_A_trianglecurv2depth_gt": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "h(f(y)) -> h(f(F(EC(a(x)))))": lambda y, y_hat, x, norm: norm(h(f(y)), h(f(F(EC(a(x)))))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)),
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))),
                "f(F(EC[a(x)]))": lambda y, y_hat, x: f(F(EC(a(x)))),
                "h(f(F(EC[a(x)])))": lambda y, y_hat, x: h(f(F(EC(a(x))))),
            }
        ),
    "wGTinflux_A_trianglecurv2depth2_gt": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "h(f(y)) -> g(F(EC(a(x))))": lambda y, y_hat, x, norm: norm(h(f(y)), g(F(EC(a(x))))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)),
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))),
                "g(F(EC[a(x)]))": lambda y, y_hat, x: g(F(EC(a(x)))),
            }
        ),
    "wGTinflux_2dkeypt_A": 
        (
            {
                "y -> F(KC(k(x)))": lambda y, y_hat, x, norm: norm(y, F(KC(k(x)))),
            },
            {
                "k(x)": lambda y, y_hat, x: k(x), 
                "KC[k(x)]": lambda y, y_hat, x: KC(k(x)),
                "F(KC[k(x)])": lambda y, y_hat, x: F(KC(k(x))),
            }
        ),
    "wGTinflux_curvature_A": 
        (
            {
                "y -> F(RC(x))": lambda y, y_hat, x, norm: norm(y, F(RC(x))),
            },
            {
                "RC(x)": lambda y, y_hat, x: RC(x), 
                "F(RC(x))": lambda y, y_hat, x: F(RC(x)),
            }
        ),
    "wGTinflux_depthAB": 
        (
            {
                "y -> G(ED(a(x)))": lambda y, y_hat, x, norm: norm(y, G(ED(a(x)))),
                "g(y) -> ED(a(x))": lambda y, y_hat, x, norm: norm(g(y), ED(a(x))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)), 
                "G(ED[a(x)])": lambda y, y_hat, x: G(ED(a(x))), 
                "g(y)": lambda y, y_hat, x: g(y), 
            }
        ),
    "wGTinflux_curvA_depthB": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "g(y) -> ED(a(x))": lambda y, y_hat, x, norm: norm(g(y), ED(a(x))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)), 
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)), 
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))), 
                "g(y)": lambda y, y_hat, x: g(y), 
            }
        ),
    "wGTinflux_curvA_depthB_trianglecurv2depth_gt": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "g(y) -> ED(a(x))": lambda y, y_hat, x, norm: norm(g(y), ED(a(x))),
                "h(f(y)) -> ED(a(x))": lambda y, y_hat, x, norm: norm(h(f(y)), ED(a(x))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)), 
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)), 
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))), 
                "f(y)": lambda y, y_hat, x: f(y), 
                "h(f(y))": lambda y, y_hat, x: h(f(y)), 
                "g(y)": lambda y, y_hat, x: g(y), 
            }
        ),
    "wGTinflux_curvA_depthB_triangledepth2curv_gt": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "g(y) -> ED(a(x))": lambda y, y_hat, x, norm: norm(g(y), ED(a(x))),
                "f(y) -> H(ED(a(x)))": lambda y, y_hat, x, norm: norm(f(y), H(ED(a(x)))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)), 
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)), 
                "H(ED[a(x)])": lambda y, y_hat, x: H(ED(a(x))), 
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))), 
                "f(y)": lambda y, y_hat, x: f(y), 
                "g(y)": lambda y, y_hat, x: g(y), 
            }
        ),
    "wGTinflux_curvA_depthI_trianglecurv2depth_gt": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "h(f(y)) -> ED(a(x))": lambda y, y_hat, x, norm: norm(h(f(y)), ED(a(x))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)), 
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)), 
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))), 
                "f(y)": lambda y, y_hat, x: f(y), 
                "h(f(y))": lambda y, y_hat, x: h(f(y)), 
            }
        ),
    "wGTinflux_curvA_depthI_triangledepth2curv_gt": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "f(y) -> H(ED(a(x)))": lambda y, y_hat, x, norm: norm(f(y), H(ED(a(x)))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)), 
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)), 
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))), 
                "H(ED[a(x)])": lambda y, y_hat, x: H(ED(a(x))), 
                "f(y)": lambda y, y_hat, x: f(y), 
            }
        ),
    "wGTinflux_curvA_depthB_2dkeyptA": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "g(y) -> ED(a(x))": lambda y, y_hat, x, norm: norm(g(y), ED(a(x))),
                "y -> F(KC(k(x)))": lambda y, y_hat, x, norm: norm(y, F(KC(k(x)))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)), 
                "ED[a(x)]": lambda y, y_hat, x: ED(a(x)), 
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))), 
                "k(x)": lambda y, y_hat, x: k(x), 
                "KC[k(x)]": lambda y, y_hat, x: KC(k(x)), 
                "F(KC[k(x)])": lambda y, y_hat, x: F(KC(k(x))), 
                "g(y)": lambda y, y_hat, x: g(y), 
            }
        ),
    "wGTinflux_curvA_2dkeyptA": 
        (
            {
                "y -> F(EC(a(x)))": lambda y, y_hat, x, norm: norm(y, F(EC(a(x)))),
                "y -> F(KC(k(x)))": lambda y, y_hat, x, norm: norm(y, F(KC(k(x)))),
            },
            {
                "a(x)": lambda y, y_hat, x: a(x), 
                "EC[a(x)]": lambda y, y_hat, x: EC(a(x)), 
                "F(EC[a(x)])": lambda y, y_hat, x: F(EC(a(x))), 
                "k(x)": lambda y, y_hat, x: k(x), 
                "KC[k(x)]": lambda y, y_hat, x: KC(k(x)), 
                "F(KC[k(x)])": lambda y, y_hat, x: F(KC(k(x))), 
            }
        ),
    "percep_imagenet": 
        (
            {
                "y -> y^": lambda y, y_hat, x, norm: norm(y, y_hat),
                "NIm(y) -> NIm(y^)": lambda y, y_hat, x, norm: norm(NIm(y), NIm(y_hat)),
            },
            {}
        ),
    "percep_random": 
        (
            {
                "y -> y^": lambda y, y_hat, x, norm: norm(y, y_hat),
                "RND(y) -> RND(y^)": lambda y, y_hat, x, norm: norm(RND(y), RND(y_hat)),
            },
            {
                "RND(y)": lambda y, y_hat, x: RND(y), 
                "RND(y^)": lambda y, y_hat, x: RND(y_hat), 
            }
        ),
    "cycle_tests": 
        (
            {
                "f(y) -> f(y^)": lambda y, y_hat, x, norm: norm(f(y), f(y_hat)),
                "F(f(y))_frozen -> y": lambda y, y_hat, x, norm: norm(F(f(y)).detach(), y),

                # "g(y) -> g(y^)": lambda y, y_hat, x, norm: norm(g(y), g(y_hat)),
                # "G(g(y))_frozen -> y": lambda y, y_hat, x, norm: norm(G(g(y)).detach(), y),

                "s(y) -> s(y^)": lambda y, y_hat, x, norm: norm(s(y), s(y_hat)),
                "S(s(y))_frozen -> y": lambda y, y_hat, x, norm: norm(S(s(y)), y),

                "nr(y) -> nr(y^)": lambda y, y_hat, x, norm: norm(nr(y), nr(y_hat)),
                "rn(nr(y)))_frozen -> y^": lambda y, y_hat, x, norm: norm(rn(nr(y)).detach(), y),

                "Nk3(y) -> Nk3(y^)": lambda y, y_hat, x, norm: norm(Nk3(y), Nk3(y_hat)),
                "k3N(Nk3(y))_frozen -> y^": lambda y, y_hat, x, norm: norm(k3N(Nk3(y)).detach(), y),
            },
            {
                "f(y)": lambda y, y_hat, x: f(y), 
                "f(y^)": lambda y, y_hat, x: f(y_hat),
                "F(f(y))": lambda y, y_hat, x: F(f(y)),

                # "g(y)": lambda y, y_hat, x: g(y), 
                # "g(y^)": lambda y, y_hat, x: g(y_hat),
                # "G(g(y))": lambda y, y_hat, x: G(g(y)),

                "s(y)": lambda y, y_hat, x: s(y), 
                "s(y^)": lambda y, y_hat, x: s(y_hat),
                "S(s(y))": lambda y, y_hat, x: S(s(y)),

                "nr(y)": lambda y, y_hat, x: nr(y), 
                "nr(y^)": lambda y, y_hat, x: nr(y_hat),
                "rn(nr(y))": lambda y, y_hat, x: rn(nr(y)),

                "Nk3(y)": lambda y, y_hat, x: Nk3(y), 
                "Nk3(y^)": lambda y, y_hat, x: Nk3(y_hat),
                "k3N(Nk3(y))": lambda y, y_hat, x: k3N(Nk3(y)),
            }
        ),

        "nontriv_cycle": 
        (
            {
                "f(y) -> f(y^)": lambda y, y_hat, x, norm: norm(f(y), f(y_hat)),
                "F(f(y))_frozen -> y": lambda y, y_hat, x, norm: norm(F(f(y)).detach(), y),

                # "g(y) -> g(y^)": lambda y, y_hat, x, norm: norm(g(y), g(y_hat)),
                # "G(g(y))_frozen -> y": lambda y, y_hat, x, norm: norm(G(g(y)).detach(), y),

                "s(y) -> s(y^)": lambda y, y_hat, x, norm: norm(s(y), s(y_hat)),
                "nr(y) -> nr(y^)": lambda y, y_hat, x, norm: norm(nr(y), nr(y_hat)),
                "Nk3(y) -> Nk3(y^)": lambda y, y_hat, x, norm: norm(Nk3(y), Nk3(y_hat)),
            },
            {
                "f(y)": lambda y, y_hat, x: f(y), 
                "f(y^)": lambda y, y_hat, x: f(y_hat),
                "F(f(y))": lambda y, y_hat, x: F(f(y)),

                # "g(y)": lambda y, y_hat, x: g(y), 
                # "g(y^)": lambda y, y_hat, x: g(y_hat),
                # "G(g(y))": lambda y, y_hat, x: G(g(y)),

                "s(y)": lambda y, y_hat, x: s(y), 
                "s(y^)": lambda y, y_hat, x: s(y_hat),
                # "S(s(y))": lambda y, y_hat, x: S(s(y)),

                "nr(y)": lambda y, y_hat, x: nr(y), 
                "nr(y^)": lambda y, y_hat, x: nr(y_hat),
                # "rn(nr(y))": lambda y, y_hat, x: rn(nr(y)),

                "Nk3(y)": lambda y, y_hat, x: Nk3(y), 
                "Nk3(y^)": lambda y, y_hat, x: Nk3(y_hat),
                # "k3N(Nk3(y))": lambda y, y_hat, x: k3N(Nk3(y)),
            }
        ),
}

### FUNCTIONAL LOSSES

class FunctionalLoss(object):
    
    def __init__(self, config=None, losses={}, metrics={}, plot_losses={}, src_task="rgb", dest_task="normal"):

        self.losses = losses
        self.plot_losses = plot_losses
        if config is not None:
            self.losses, self.plot_losses = loss_configs[config]
        self.loss_names = sorted(self.losses.keys())
        self.src_task, self.dest_task = TASK_MAP[src_task], TASK_MAP[dest_task]

    def __call__(self, y, y_hat, x):
        y.parents = [n]
        loss_values = [self.losses[loss](y, y_hat, x, self.dest_task.norm) for loss in self.loss_names]
        return sum(loss_values), [loss.detach() for loss in loss_values]

    def logger_hooks(self, logger):
        for loss in self.losses.keys():
            logger.add_hook(partial(jointplot, logger=logger, loss_type=f"{loss}"), feature=f"val_{loss}", freq=1)

    def logger_update(self, logger, train_metrics, val_metrics):

        for loss, metric in zip(self.losses.keys(), train_metrics):
            logger.update(f"train_{loss}", np.mean(metric))

        for loss, metric in zip(self.losses.keys(), val_metrics):
            logger.update(f"val_{loss}", np.mean(metric))


class NormalizedFunctionalLoss(FunctionalLoss):
    
    def __init__(self, config=None, losses={}, plot_losses={}, update_freq=1):
        super().__init__(config=config, losses=losses, plot_losses=plot_losses)
        
        self.loss_coeffs = None
        self.update_freq = update_freq
        self.step_count = 0

    def __call__(self, y, y_hat, x):
        y.parents = [n]
        loss_values = [self.losses[loss](y, y_hat, x, self.norm) for loss in self.loss_names]

        if self.step_count % self.update_freq == 0:
            self.loss_coeffs = [1/loss.detach() for loss in loss_values]
        self.step_count += 1

        final_loss = sum(c*loss for loss, c in zip(loss_values, self.loss_coeffs))
        return final_loss, [loss.detach() for loss in loss_values]


class MixingFunctionalLoss(FunctionalLoss):
    
    def __init__(self, config=None, losses={}, plot_losses={}, update_freq=1):
        super().__init(config=config, losses=losses, plot_losses=plot_losses)
        
        if len(losses) != 2:
            raise Exception("MixingFunctionalLoss requires exactly two loss functions")

        print ("WARNING: Changing all functional models to use checkpoint=False")
        for model in functional_transfers:
            model.checkpoint = False

    def __call__(self, y, y_hat, x):
        y.parents = [n]
        loss_values = [self.losses[loss](y, y_hat, x, self.norm) for loss in self.loss_names]
        c1, c2 = 1, 1
        if y.requires_grad:
            c1, c2 = calculate_weight(model, loss_values[0], loss_values[1])

        return c1*loss_values[0] + c2*loss_values[1], [loss.detach() for loss in loss_values] + [c1]

    def logger_hooks(self, logger):
        super().logger_hooks(logger)
        logger.add_hook(partial(jointplot, logger=logger, loss_type="c"), feature=f"val_c", freq=1)

    def logger_update(self, logger, train_metrics, val_metrics):
        super().logger_update(logger, train_metrics[:-1], val_metrics[:-1])
        logger.update(f"train_c", np.mean(train_metrics[-1]))
        logger.update(f"val_c", np.mean(val_metrics[-1]))


class CurriculumFunctionalLoss(FunctionalLoss):
    
    def __init__(self, config=None, losses={}, plot_losses={}, initial_coeffs=[1.0, 0.0], step=[0.0, 0.1]):
        super().__init__(config=config, losses=losses, plot_losses=plot_losses)
        
        self.loss_coeffs = initial_coeffs
        self.step = step

    def __call__(self, y, y_hat, x):
        y.parents = [n]
        loss_values = [self.losses[loss](y, y_hat, x, self.norm) for loss in self.loss_names]
        final_loss = sum(c*loss for loss, c in zip(loss_values, self.loss_coeffs))
        return final_loss, [loss.detach() for loss in loss_values]

    def logger_update(self, logger, train_metrics, val_metrics):
        super().logger_update(logger, train_metrics, val_metrics)
        self.loss_coeffs = [loss + step for loss, step in zip(self.loss_coeffs, self.step)]
        print ("Updating loss coefficients: ", self.loss_coeffs)

