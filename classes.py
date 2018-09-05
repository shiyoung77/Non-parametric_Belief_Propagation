import numpy as np
import collections

def wrap2pi(theta):
    while theta >= np.pi:
        theta -= 2 * np.pi
    while theta < -np.pi:
        theta += 2 * np.pi
    return theta


class Message(object):
    def __init__(self, gaussians=None, weights=None):
        if gaussians is None:
            self.gaussians = []
        if weights is None:
            self.weights = np.zeros([])

    def __len__(self):
        return len(self.gaussians)

    def __getitem__(self, idx):
        return self.gaussians[idx], self.weights[idx]


class Gaussian(object):
    def __init__(self, x, y, theta, w, h, x_var, y_var, theta_var, w_var, h_var):
        # mean
        self.x = x
        self.y = y
        self.theta = wrap2pi(theta)
        self.w = w
        self.h = h

        # stddev
        eps = 1e-6
        self.x_var = x_var + eps
        self.y_var = y_var + eps
        self.theta_var = theta_var + eps
        self.w_var = w_var + eps
        self.h_var = h_var + eps

        # KEY ASSUMPTION: The covariance matrix is a DIAGNOAL matrix
        self.mean = np.array([self.x, self.y, self.theta, self.w, self.h])
        self.cov = np.diag([self.x_var, self.y_var, self.theta_var, self.w_var, self.h_var])

    def __str__(self):
        return "[mean] x:{0}, y:{1}, theta:{2}, w:{3}, h:{4}\n \
                [stddev] x:{5}, y:{6}, theta:{7}, w:{8}, h:{9}".format(self.x,
                        self.y, self.theta, self.w, self.h, self.x_var, self.y_var,
                        self.theta_var, self.w_var, self.h_var)

    def __repr__(self):
        return self.__str__()


class Sample(object):
    def __init__(self, x, y, theta, w, h):
        self.__x = x
        self.__y = y
        self.__theta = wrap2pi(theta)
        self.__w = w
        self.__h = h
        self.properties = (self.__x, self.__y, self.__theta, self.__w, self.__h)

    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def theta(self):
        return self.__theta
    
    @property
    def w(self):
        return self.__w
    
    @property
    def h(self):
        return self.__h
    
    def __str__(self):
        return "x:{0}, y:{1}, theta:{2}, w:{3}, h:{4}".format(
            self.__x, self.__y, self.__theta, self.__w, self.__h)

    def __repr__(self):
        return self.__str__()
