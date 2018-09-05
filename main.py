import time
import numpy as np
from matplotlib import pyplot as plt

from draw import *
from initialize import *
from gibbs_sampler import *
from classes import *
from sample_neighbor import *

if __name__=="__main__":
    num_iterations = 30
    num_components = 40
    Gibbs_iterations = 40
    num_samples_for_belief_updating = 30
    dropout = 0.3
    keep_num = int((1 - dropout) * num_components)
    drop_num = num_components - keep_num

    # Load observation
    observation, h, w = loadObservation()

    # global parameters
    rad = 15 # circle
    w = 84   # links width
    h = 12   # links height
    init_angle = np.pi / 18
    sigma_p = 5
    sigma_th = 10*np.pi/180
    sigma_s = 2
    eps = 1e-5

    ## Initialize graph
    [pairs, node_ids, neighbor_dict, num_pairs] = initGraph()

    # initialize a message for each node
    displayImage_msg = observation.copy()
    msg_prev = dict() 
    for pair in pairs:
        msg_prev[pair], gt_circle, gt_links = initialize(pair[1], num_components, init_angle, w, h)
        displayMessages(msg_prev[pair], labelFromId(pair[1]), displayImage_msg)
    cv2.imwrite("initialize.jpg", displayImage_msg)

    # iterate to converge
    for i in range(num_iterations):
        displayImage_msg = observation.copy()

        # NBP update of nonparametric message t -> s
        msg_curr = {}
        for pair in pairs:
            print ("iter: {0}, pair:".format(i), pair, flush=True)
            no_u = False
            t_id = pair[0] # t node
            s_id = pair[1] # s node
            t_label = labelFromId(t_id)
            s_label = labelFromId(s_id)
            t_neighbors = neighbor_dict[t_id]

            print("t:", t_id, ", s:", s_id, ", t_label:", t_label, ", s_label", s_label)
            print("u_id:", [u for u in t_neighbors if u != s_id])

            # step 0
            # input msg list m(u->t), u is t's neighbor exclude s
            msg_list = []
            for u_id in t_neighbors:
                if u_id != s_id:
                    msg_list.append(msg_prev[(u_id, t_id)])

            # if t has no neighbor excluding s
            if not msg_list:
                # msg_curr[pair] = msg_prev[pair]
                # continue
                # no_u = True
                msg_list.append(msg_prev[(s_id, t_id)])

            # step 1 is not used in this implementation

            # step 2
            # draw num_components samples using Gibbs sampler
            t_samples = []
            t_weights = []
            for m in range(num_components):
                start_time = time.time()
                if len(msg_list) > 1:
                    t, t_weight = gibbs_sampler(msg_list, t_label, observation, iteration=Gibbs_iterations)
                elif len(msg_list) == 1:
                    t, t_weight = direct_sampler(msg_list[0], t_label, observation)
                t_samples.append(t)
                t_weights.append(t_weight)
                print("gibbs_sampler time: {}".format(time.time() - start_time))
                print("m: {}".format(m), end=' ', flush=True)
                print(t.x, t.y, t.theta, t.w, t.h)

            # if no_u:
            #     for _ in range(keep_num):
            #         idx = np.random.choice(num_components, p=msg_prev[pair].weights)
            #         t_samples.append(Sample(*msg_prev[pair].gaussians[idx].mean))
            #         t_weights.append(msg_prev[pair].weights[idx])
            # assert len(t_weights) == num_components and len(t_samples) == num_components

            # step 3
            # for each t, sample s
            s_samples = []
            s_weights = t_weights.copy() / np.sum(t_weights)
            for t in t_samples:
                s = sample_neighbor(t, s_label, t_label, rad, w, h, sigma_s, sigma_p, sigma_th)
                s_samples.append(s)
                print(s.x, s.y, s.theta, s.w, s.h)

            samples_for_var = []
            for _ in range(1000):
                s = sample_neighbor(t, s_label, t_label, rad, w, h, sigma_s, sigma_p, sigma_th)
                samples_for_var.append(s)

            # step 4, construct msg t -> s, choose variance using kernel density estimation(KDE)
            thetas = [s.theta for s in samples_for_var]
            if s_id == 2 or s_id == 6:
                for k in range(1000):
                    if thetas[k] < 0:
                        thetas[k] += 2 * np.pi
            x_var = KDE_rule_of_thumbs([s.x for s in samples_for_var]) + eps
            y_var = KDE_rule_of_thumbs([s.y for s in samples_for_var]) + eps
            theta_var = KDE_rule_of_thumbs(thetas) + eps
            w_var = KDE_rule_of_thumbs([s.w for s in samples_for_var]) + eps
            h_var = KDE_rule_of_thumbs([s.h for s in samples_for_var]) + eps
            print(x_var, y_var, theta_var, w_var, h_var)
            print()
            assert x_var > 0 and y_var > 0 and theta_var > 0 and w_var > 0 and h_var > 0

            msg = Message()
            msg.weights = np.zeros(len(s_samples))
            for idx, s in enumerate(s_samples):
                msg.weights[idx] = s_weights[idx]
                msg.gaussians.append(Gaussian(s.x, s.y, s.theta, s.w, s.h,
                                              x_var, y_var, theta_var, w_var, h_var))
            msg_curr[pair] = msg

        for pair in pairs:
            displayMessages(msg_curr[pair], labelFromId(pair[1]), displayImage_msg)
        msg_prev = msg_curr.copy()

        # belief update, update the belief of s using all its neighbors t
        belief_img = observation.copy()
        max_belief_img = observation.copy()
        for s_id in range(1, 10):
            s_label = labelFromId(s_id)
            s_neighbors = neighbor_dict[s_id]
            print("s_id:{0}, s_label:{1}".format(s_id, s_label))
            print("s_neighbors", s_neighbors)

            # collect all the messages from t -> s 
            msg_list = []
            for pair in pairs:
                if pair[1] == s_id:
                    msg_list.append(msg_curr[pair])

            # draw N samples using Gibbs sampler and pick the one with highest weight as belief
            s_samples = []
            s_weights = []
            for n in range(num_samples_for_belief_updating):
                if len(msg_list) > 1:
                    s, s_weight = gibbs_sampler(msg_list, s_label, observation, iteration=Gibbs_iterations)
                elif len(msg_list) == 1:
                    s, s_weight = direct_sampler(msg_list[0], s_label, observation)
                s_samples.append(s)
                s_weights.append(s_weight)
                draw_sample(s, s_id, belief_img)

            if sum(s_weights) == 0:
                idx = np.random.randint(num_samples_for_belief_updating)
            else:
                idx = np.argmax(s_weights)

            s_belief = s_samples[idx]
            draw_sample(s_belief, s_id, max_belief_img)
        concat_img = np.concatenate((displayImage_msg, belief_img, max_belief_img), axis=1)
        cv2.imwrite("iter{0}.jpg".format(i), concat_img)

    # i = 1
    # print(pairs[i])
    # msg_prev[pairs[i]], gt_circle, gt_links = initialize(pairs[i][1], num_components, init_angle, w, h)
    # displayMessages(msg_prev[pairs[i]], labelFromId(pairs[i][1]), displayImage_msg)

    # plt.figure()
    # plt.imshow(displayImage_msg)
    # plt.show()
