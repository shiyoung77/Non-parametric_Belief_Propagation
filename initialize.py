import numpy as np
import scipy.io as sio
from draw import *
from classes import *

def labelFromId(num):
    if num == 9:
        return 'circle'
    return 'link_' + str(num)


def loadAndSample(object_label, num_components, init_angle, w, h, gt_links, gt_circles):
    msg = Message()
    msg.weights = np.ones(num_components) / num_components

    # stddev
    x_sigma = 1
    y_sigma = 1
    theta_sigma = np.pi / 18
    w_sigma = 0.1
    h_sigma = 0.1
    rad = 15

    # read .mat file and sample points
    if object_label.startswith("circle"):
        mat = sio.loadmat("./rc_circle.mat")
        idxs = np.random.choice(len(mat['rc_circle']), num_components)
        for i in range(num_components):
            x = mat['rc_circle'][idxs[i], 1]
            y = mat['rc_circle'][idxs[i], 0]
            # x = gt_circles[0]
            # y = gt_circles[1]
            msg.gaussians.append(Gaussian(x, y, 0, rad, rad,
                x_sigma**2, y_sigma**2, theta_sigma**2, w_sigma**2, h_sigma**2))
    else:
        link_id = int(object_label.split('_')[1])
        mat_file = "./rc_link_{0}.mat".format((link_id - 1) % 4 + 1)
        mat = sio.loadmat(mat_file)
        yx_array = mat["rc_link_{0}".format((link_id - 1) % 4 + 1)]
        idxs = np.random.choice(len(yx_array), num_components)
        for i in range(num_components):
            x = yx_array[idxs[i], 1]
            y = yx_array[idxs[i], 0]
            theta = link_id * np.pi/2 + init_angle - 2 * init_angle * np.random.rand()
            # x = gt_links[link_id-1, 0]
            # y = gt_links[link_id-1, 1]
            # theta = link_id * np.pi/2 + gt_links[link_id-1, 2]
            msg.gaussians.append(Gaussian(x, y, theta, w, h,
                x_sigma**2, y_sigma**2, theta_sigma**2, w_sigma**2, h_sigma**2))
    return msg


def initialize(object_id, num_components, init_angle, w, h):
    # gt means gronund truth
    gt_circles = np.array([314, 211, 0])
    gt_links = np.array([[312.000000, 211.000000, 20*np.pi/180],
                         [313.000000, 209.000000, -10*np.pi/180],
                         [313.000000, 209.000000, 0],
                         [317.000000, 212.000000, -10*np.pi/180],
                         [287.000000, 287.000000, 0],
                         [233.000000, 231.000000, +10*np.pi/180],
                         [310.000000, 127.000000, -20*np.pi/180],
                         [395.000000, 192.000000, 0]])
    msg = loadAndSample(labelFromId(object_id), num_components, init_angle, w, h, gt_links, gt_circles)
    return msg, gt_circles, gt_links


def initGraph():
    pairs = ((9, 1), (9, 2), (9, 3), (9, 4), (1, 9), (2, 9), (3, 9), (4, 9),
             (1, 5), (5, 1), (2, 6), (6, 2), (3, 7), (7, 3), (4, 8), (8, 4))
    node_ids = list(range(1, 10))
    neighbor_dict = dict()
    neighbor_dict[9] = (1, 2, 3, 4)
    neighbor_dict[1] = (5, 9)
    neighbor_dict[2] = (6, 9)
    neighbor_dict[3] = (7, 9)
    neighbor_dict[4] = (8, 9)
    neighbor_dict[5] = (1,)
    neighbor_dict[6] = (2,)
    neighbor_dict[7] = (3,)
    neighbor_dict[8] = (4,)
    return pairs, node_ids, neighbor_dict, len(pairs)


def displayMessages(msg, object_label, res_image):
    color_list = ['green', 'yellow', 'blue', 'red', 'green', 'yellow', 'blue', 'red']
    color_code = [[0,125,0], [125,125,0], [0,0,125], [125,0,0], [0,255,0],
                  [255,255,0], [0,0,255], [255,0,0], [255,0,255]]
    if object_label.startswith('circle'):
        for (g, w) in msg:
            draw_circle(res_image, g.x, g.y, g.h, color=color_code[8])
    else:
        link_id = int(object_label.split('_')[1])
        for (g, w) in msg:
            draw_links(res_image, g.x, g.y, g.w, g.h, g.theta, color_code[link_id - 1])

def draw_sample(sample, object_id, res_image):
    color_code = [[0,125,0], [125,125,0], [0,0,125], [125,0,0], [0,255,0],
                  [255,255,0], [0,0,255], [255,0,0], [255,0,255]]
    if object_id == 9:
        draw_circle(res_image, sample.x, sample.y, sample.h, color=color_code[object_id-1])
    else:
        draw_links(res_image, sample.x, sample.y, sample.w, sample.h, sample.theta, color_code[object_id-1])



if __name__=="__main__":
    initialize(5, 100, np.pi / 18, 84, 12)
