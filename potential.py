import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from draw import draw_circle, draw_links
from initialize import *
from weightFromNeighbor import *
from gibbs_sampler import *

def pairwise_potential(t, s, t_label, s_label):
    assert t_lable != s_label
    global sigma_s, sigma_p, sigma_th, w, h, rad, scale_neigh
    C = 7;
    del_w = 28 / 5;
    del_h = 4 / 5;
    scale = scale_neigh.copy();
    SIG_n = np.diag([sigma_p**2, sigma_p**2, sigma_th**2])
    SIG_n_spl = np.diag([sigma_p**2 * 4, sigma_p**2 * 4, sigma_th**2 / 4])
    eps = 1e-7

    # either t or s is the circle node
    if t_label == 'circle':
        p_t = Sample(s.x,
                     s.y,
                     0,
                     0.5*(s.w/del_w, s.h/del_h),
                     t.w) #??
        val = mvn.pdf([p_t.x, p_t.y], [t.x, t.y], scale[0:2].dot(SIG_n[0:2, 0:2])) / \
                mvn.pdf([t.x, t.y], [t.x, t.y], scale[0:2].dot(SIG_n[0:2, 0:2]));            
        return val + eps

    if s_label == 'circle':
        t_id = int(t_label.split('_')[1])
        p_t = Sample(s.x,
                     s.y,
                     t_id * pi / 2,
                     2 * C * s.w * del_w * del_h / (C*del_h + del_w),
                     2 * s.w * del_w * del_h / (C*del_h + del_w))
        val = mvn.pdf([p_t.x, p_t.y, p_t.theta], [t.x, t.y, t.theta], scale.dot(SIG_n)) / \
                mvn.pdf([t.x, t.y, t.theta], [t.x, t.y, t.theta], scale.dot(SIG_n))
        return val + eps
    
    # neither t nor s is the circle node
    t_id = int(t_label.split('_')[1])
    s_id = int(s_label.split('_')[1])
    if t_id == s_id + 4:
        p_t = Sample(t.x + t.w*np.cos(t.theta),
                     t.y + t.w*np.sin(t.theta),
                     t.theta,
                     s.w, #??
                     s.h) #??
        val = mvn.pdf([p_t.x, p_t.y, p_t.theta], [s.x, s.y, s.theta], SIG_n_spl) / \
                mvn.pdf([s.x, s.y, s.theta], [s.x, s.y, s.theta], SIG_n_spl) 
        return val + eps

    if s_id == t_id + 4:
        p_t = Sample(s.x + s.w*np.cos(s.theta),
                     s.y + s.w*np.sin(s.theta),
                     s.theta,
                     s.w, #??
                     s.h) #??
        val = mvn.pdf([p_t.x, p_t.y, p_t.theta], [t.x, t.y, t.theta], scale.dot(SIG_n)) / \
                mvn.pdf([t.x, t.y, t.theta], [t.x, t.y, t.theta], scale.dot(SIG_n))
        return val + eps
    return eps


def prior_from_u(msg_ut, current_id, neighbor_id, s):
    current_label = labelFromId(current_id)
    neighbor_label = labelFromId(neighbor_id)
    weight_prior = 0
    for t, weight in msg_ut:
        weight_prior += weight * pairwisePotential(s, t, current_label, neighbor_label)
    return weight_prior


def get_neighbors(s_id, t_id, neighbor_dict):
    neighbors = []
    for neighbor_id in neighbor_dict[t]:
        if neighbor_id != s_id and neighbor_id != t_id:
            neighbors.append(neighbor_id)
    return neighbors



def weight_from_neighbor(msgs, msg_prev, current_id, neighbor_id, observation, neighbor_dict):
    global W, H, rad, w, h, sigma_s, sigma_p, sigma_th
    neighbor_label = labelFromId(neighbor_id)
    current_label = labelFromId(current_id)

    # Sample for neighbours for every position in msg_positions
    msg = msgs[(neighbor_id, current_id)]
    M = len(msg)
    weights_unary = np.zeros(M)
    weights_prior = np.zeros(M)
    weights = np.zeros(M)

    for i, (s, _) in enumerate(msg):
        t = sample_neighbor(s, neighbor_label, current_label, rad, w, h, sigma_s, sigma_p, sigma_th)
        if t.x < 0 and t.y < 0 and t.x >= W and t.y >= H:
            continue
        weights_unary[i] = unary_potential(neighbor_label, t.x, t.y, t.theta, t.w, t.h, observation)
        neighbors = get_neighbors(current_id, neighbor_id, neighbor_dict)
        if neighbors:
            weights_from_priors = np.zeros(len(neighbors))
            for idx, u_id in enumerate(neighbors):
                weights_from_priors[idx] = prior_from_u(msg_prev[(u_id, neighbor_id)], current_id, neighbor_id, s)
            weights_prior[i] = np.prod(weights_from_priors)

    weights_unary_sum = np.sum(weights_unary)
    weights_prior_sum = np.sum(weights_prior)
    if weights_unary_sum != 0:
        weights_unary = weights_unary / weights_unary_sum
    else:
        weights_unary = np.ones(M) / M

    if weights_prior_sum != 0:
        weights_prior = weights_prior / weights_prior_sum
    else:
        weights_prior = np.ones(M) / M

    return weights_unary * weights_prior


