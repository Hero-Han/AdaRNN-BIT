from loss.nadv_loss import adv
from loss.ncoral import CORAL
from loss.ncos import cosine
from loss.nkl_js import kl_div, js
from loss.nmmd import MMD_loss
from loss.nmutual_info import Mine
from loss.pair_dist import pairwise_dist

__all__ = [
    'adv',
    'CORAL',
    'cosine',
    'kl_div',
    'js'
    'MMD_loss',
    'Mine',
    'pairwise_dist'
]