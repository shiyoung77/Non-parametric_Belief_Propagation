import numpy as np
import statistics
from scipy.stats import multivariate_normal
from scipy.stats import norm
from numpy.linalg import pinv
from functools import reduce
from classes import *
from draw import *

def gaussian_product(d1, d2):
    """
    sigma3 = (sigma1^(-1) + sigma2^(-1))^(-1)
    mu3 = sigma3*sigma1^(-1)*mu1 + sigma3*sigma2^(-1)*mu2

    The scaling factor is NOT calculated in this implementation

    reference:
    https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions
    """
    assert isinstance(d1, Gaussian) and isinstance(d2, Gaussian)
    inv_cov1 = pinv(d1.cov)
    inv_cov2 = pinv(d2.cov)
    sigma = pinv(inv_cov1 + inv_cov2)
    mu = sigma @ (inv_cov1 @ d1.mean + inv_cov2 @ d2.mean)
    return Gaussian(*mu, *np.diag(sigma))


def gaussian_product_diag(d1, d2):
    """
    same as gaussian_product, but add a key assumption
    Assumption: the covariance of d1 and d2 are diagonal matrix
    """
    inv_cov1 = 1 / np.diag(d1.cov)
    inv_cov2 = 1 / np.diag(d2.cov)
    sigma = 1 / (inv_cov1 + inv_cov2)
    mu = sigma * (inv_cov1*d1.mean + inv_cov2*d2.mean) 
    return Gaussian(*mu, *sigma)


def unary_potential(object_label, sample, observation):
    x, y, theta, w, h = sample.properties
    x = int(round(x))
    y = int(round(y))

    if object_label == 'circle':
        w = int(np.ceil(w))
        h = int(np.ceil(h))
        img = np.zeros((w * 2, w * 2, 3), np.uint8)
    else:
        w_c = int(np.ceil(w / 2))
        img = np.zeros((w_c, w_c, 3), np.uint8)

    if object_label == 'circle':
        draw_circle(img, w, w, w, [255, 0, 0])
        img = img[:, :, 0].astype(np.double) / 255
        img[img > 0] = 1
        img_r = img.reshape((-1, 1))

        y1 = y - w + 1
        y2 = y + w
        x1 = x - w + 1
        x2 = x + w
        if y1 < 0 or y2 >= observation.shape[1] or x1 < 0 or x2 >= observation.shape[0]:
            val = 0
        else:
            sub_observation = observation[y1:y2+1, x1:x2+1, 0].astype(np.double) / 255
            sub_observation[sub_observation > 0] = 1
            sub_r = sub_observation.reshape((-1, 1))
            idx = (img_r == 1)
            val = np.sum(img_r[idx] == sub_r[idx]) / max(np.sum(sub_r == 1), np.sum(img_r == 1))

    else:
        link_id = int(object_label.split('_')[1])
        x_bar = int(round(x + 0.75*w*np.cos(theta)))
        y_bar = int(round(y + 0.75*w*np.sin(theta)))
        draw_links(img, w/4, w/4, w, h, theta, [255, 0, 0], False)
        img = img[:, :, 0].astype(np.double) / 255
        img[img < 0.5] = 0
        img[img > 0] = 1
        img_r = img.reshape((-1, 1))

        offset = 1 if w_c % 2 == 0 else 0
        half_w_c = w_c // 2
        y1 = y_bar - half_w_c + offset
        y2 = y_bar + half_w_c
        x1 = x_bar - half_w_c + offset
        x2 = x_bar + half_w_c

        if y1 < 0 or y2 >= observation.shape[1] or x1 < 0 or x2 >= observation.shape[0]:
            val = 0
        else:
            sub_observation = observation[y1:y2+1, x1:x2+1, 0].astype(np.double) / 255
            sub_observation[sub_observation < 0.5] = 0
            sub_observation[sub_observation > 0] = 1
            sub_r = sub_observation.reshape((-1, 1))
            idx = (img_r == 1)
            val = np.sum(img_r[idx] == sub_r[idx]) / max(np.sum(sub_r == 1), np.sum(img_r == 1))

    return val + 1e-8


def loadObservation(filename=None):
    if filename is None:
        obs_img = cv2.imread('./no_clutter_pampas.png')
    else:
        obs_img = cv2.imread(filename)
    obs_img = cv2.cvtColor(obs_img, cv2.COLOR_BGR2RGB)
    return obs_img, obs_img.shape[0], obs_img.shape[1]


def test_unary(node_id):
    rad = 15
    w = 84
    h = 12
    gt_links = np.array([[312.000000, 211.000000, 0],
                         [313.000000, 209.000000, 0],
                         [313.000000, 209.000000, 0],
                         [317.000000, 212.000000, 0],
                         [287.000000, 287.000000, 0],
                         [233.000000, 231.000000, 0],
                         [310.000000, 127.000000, 0],
                         [395.000000, 192.000000, 0]])

    [observation, W, H] = loadObservation()
    heat_map = np.zeros((W, H))
    step = 2;
    if node_id == 9:
        object_label = 'circle'
        theta = 0
        w = rad
        h = rad
    else:
        object_label = 'link_' + str(node_id)
        theta = node_id*np.pi/2 + gt_links[node_id-1, 2]

    max_potential = 0
    for i in range(0, W, step):
        for j in range(0, H, step):
            sample = Sample(i, j, theta, w, h)
            potential = unary_potential(object_label, sample, observation)
            heat_map[j, i] = potential
            max_potential = max(max_potential, potential)
    print("max potential:", max_potential)
    return heat_map


def direct_sampler(msg, object_label, observation):
    idx = np.random.choice(len(msg), p=msg.weights)
    gaussian, weight = msg[idx]
    properties = np.random.multivariate_normal(gaussian.mean, gaussian.cov)
    sample = Sample(*properties)
    weight *= unary_potential(object_label, sample, observation)
    return sample, weight


def gibbs_sampler(msg_list, object_label, observation, iteration=10, verbose=False):
    if not (isinstance(msg_list, list) or isinstance(msg_list, tuple)):
        raise AssertionError("msg_list must be a list or a tuple")

    if len(msg_list) < 2:
        raise AssertionError("length of the msg_list must be at least 2")

    M = len(msg_list[0])

    # step 1, initialize weights and pick a gaussian for each msg
    gaussians = []
    for msg in msg_list:
        label = np.random.choice(M, p=msg.weights)
        gaussians.append(msg.gaussians[label])
    
    # step 2, iteration
    for k in range(iteration):
        for j, msg in enumerate(msg_list):
            # step (a)
            # compute the unscaled product of picked gaussians in other messages
            others_product = reduce(gaussian_product_diag, gaussians[:j] + gaussians[j+1:])

            # step (b)
            # pick the label for current message
            weights = np.zeros(M)
            for i in range(M):
                gaussian_bar = gaussian_product_diag(msg.gaussians[i], others_product)

                # choose the convenient x as the mean of gaussian_bar
                x = gaussian_bar.mean
                w1 = multivariate_normal.pdf(x, mean=msg.gaussians[i].mean, cov=msg.gaussians[i].cov)
                w2 = multivariate_normal.pdf(x, mean=others_product.mean, cov=others_product.cov)
                w3 = multivariate_normal.pdf(x, mean=gaussian_bar.mean, cov=gaussian_bar.cov)
                w4 = unary_potential(object_label, Sample(*x), observation)
                weights[i] = msg.weights[i] * w1 * w2 / w3 * w4
            
            if np.sum(weights) != 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(M) / M

            label = np.random.choice(M, p=weights)
            gaussians[j] = msg.gaussians[label]

    # calculate the products of all the picked gaussians
    product = reduce(gaussian_product_diag, gaussians)

    # step (4)
    # sample a number from the gaussian product
    properties = np.random.multivariate_normal(product.mean, product.cov)
    sample = Sample(*properties)

    # step (5)
    # assign importance weight
    sample_weight = unary_potential(object_label, sample, observation) / \
                        unary_potential(object_label, Sample(*product.mean), observation)
    return sample, sample_weight


def KDE_rule_of_thumbs(samples):
    '''
        https://en.wikipedia.org/wiki/Kernel_density_estimation
    '''
    sigma = statistics.stdev(samples)
    return (4 / 3 * sigma**5 / len(samples))**0.2


if __name__=='__main__':
    heat_map = test_unary(9)
    plt.imshow(heat_map)
    plt.show()
