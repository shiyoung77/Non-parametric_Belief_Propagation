import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_circle(img, x, y, r, color='m', grad=1):
    color_map = {'r':[255,0,0], 'b':[0,0,255], 'g':[0,255,0], 'w':[255,255,255], 'b':[0,0,0], 'm':[255,0,255]}
    if isinstance(color, str):
        color = color_map[color]
    x = int(round(x))
    y = int(round(y))
    r = int(round(r))
    cv2.circle(img, (x, y), r, color * grad, -1)


def draw_links(img, x, y, w, h, angle, color='r', flag=True):
    color_map = {'r':[255,0,0], 'b':[0,0,255], 'g':[0,255,0], 'w':[255,255,255], 'b':[0,0,0], 'm':[255,0,255]}
    if isinstance(color, str):
        color = color_map[color]
    if flag:
        r1 = np.sqrt(w*w/4 + h*h/4)
        r2 = np.sqrt(w*w + h*h/4)
        theta1 = np.arctan(h/w)
        theta2 = np.arctan(h/(2*w))

        tl = [x + r1*np.cos(angle + theta1), y + r1*np.sin(angle + theta1)]
        tr = [x + r2*np.cos(angle + theta2), y + r2*np.sin(angle + theta2)]
        br = [x + r2*np.cos(angle - theta2), y + r2*np.sin(angle - theta2)]
        bl = [x + r1*np.cos(angle - theta1), y + r1*np.sin(angle - theta1)]
        pts = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))

        # polylines(img, pts, isClosed, color, thickness)
        cv2.polylines(img, [pts], True, color, thickness=2);
        cv2.fillConvexPoly(img, pts, color)

        # circle(img, center, radius, color, thinkness(negative=filled))
        x = int(round(x))
        y = int(round(y))
        cv2.circle(img, (x, y), 3, color, thickness=-1)
    else:
        w /= 2
        r = np.sqrt(w*w/4 + h*h/4)
        theta = np.arctan(h/w)
        tl = [x + r*np.cos(angle + np.pi - theta), y + r*np.sin(angle + np.pi - theta)]
        tr = [x + r*np.cos(angle + theta)        , y + r*np.sin(angle + theta)]
        br = [x + r*np.cos(angle - theta)        , y + r*np.sin(angle - theta)]
        bl = [x + r*np.cos(angle + np.pi + theta), y + r*np.sin(angle + np.pi + theta)]
        pts = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, thickness=2);
        cv2.fillConvexPoly(img, pts, color)

if __name__=='__main__':
    img = np.zeros((512, 512, 3), np.uint8)
    draw_links(img, 300, 300, 100, 100, np.pi / 6, 'g', True)
    draw_circle(img, 100, 100, 10)
    draw_circle(img, 100, 300, 10, 'g')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
