import numpy as np
from classes import *
from draw import *

def sample_neighbor(t, s_label, t_label, rad, w, h, sigma_s, sigma_p, sigma_th):
    # Given object_t, sample for the object_s
    C = 7
    del_w = 28 / 5
    del_h = 4 / 5
    SIG_n = np.diag([sigma_p**2, sigma_p**2, sigma_th**2])
    SIG_n_spl = np.diag([(2*sigma_p)**2, (2*sigma_p)**2, (sigma_th/2)**2])

    if s_label == 'circle':
        return Sample(t.x + np.random.normal(0, sigma_p),
                      t.y + np.random.normal(0, sigma_p),
                      0 + np.random.normal(0, sigma_th),
                      0.5*(t.w/del_w + t.h/del_h),# + np.random.normal(0, sigma_s),
                      0.5*(t.w/del_w + t.h/del_h))# + np.random.normal(0, sigma_s))

    if t_label == 'circle':
        t_id = int(s_label.split('_')[1])
        return Sample(t.x + np.random.normal(0, sigma_p),
                      t.y + np.random.normal(0, sigma_p),
                      t_id*np.pi/2 + np.random.normal(0, sigma_th),
                      2*C*t.w*del_w*del_h/(C*del_h+del_w),# + np.random.normal(0, sigma_s),
                      2*t.w*del_w*del_h/(C*del_h+del_w))# + np.random.normal(0, sigma_s))

    s_id = int(s_label.split('_')[1])
    t_id = int(t_label.split('_')[1])

    if t_id == s_id + 4:
        noise_x = t.x + np.random.normal(0, sigma_p)
        noise_y = t.y + np.random.normal(0, sigma_p)
        noise_theta = t.theta + np.pi + np.random.normal(0, sigma_th/2)
        return Sample(noise_x + t.w*np.cos(noise_theta),
                      noise_y + t.w*np.sin(noise_theta),
                      noise_theta - np.pi,
                      t.w,# + np.random.normal(0, sigma_s),
                      t.h)# + np.random.normal(0, sigma_s))

    if s_id == t_id + 4:
        return Sample(t.x + t.w*np.cos(t.theta) + np.random.normal(0, sigma_p),
                      t.y + t.w*np.sin(t.theta) + np.random.normal(0, sigma_p),
                      t.theta + np.random.normal(0, sigma_th),
                      t.w,# + np.random.normal(0, sigma_s),
                      t.h)# + np.random.normal(0, sigma_s))


def test_pairwise_sampling(node_id):
    global rad, w, h, sigma_s, sigma_p, sigma_th;
    gt_circle = np.array([314, 211])
    gt_links = np.array([[312.000000, 211.000000, 20*np.pi/180],
                         [313.000000, 209.000000, -10*np.pi/180],
                         [313.000000, 209.000000, 0],
                         [317.000000, 212.000000, -10*np.pi/180],
                         [287.000000, 287.000000, 0],
                         [233.000000, 231.000000, +10*np.pi/180],
                         [310.000000, 127.000000, -20*np.pi/180],
                         [395.000000, 192.000000, 0]])

    sigma_p = 5
    sigma_th = 5*np.pi/180
    sigma_s = 2

    rad = 15 # circle
    w = 84   # links width
    h = 12   # links height      
    load_img = cv2.imread('./star_pampas.png')
    img1 = load_img.copy()
    img2 = load_img.copy()
    img3 = load_img.copy()
    img4 = load_img.copy()
    N = 100
    green = [0, 255, 0]

    fig, axes = plt.subplots(nrows=2, ncols=2)

    # img1
    title = 'Given circle location (red) the inner link samples for node ' + str(node_id)
    t = Sample(gt_circle[0], gt_circle[1], 0, rad, rad)
    s_label = 'link_' + str(node_id)
    t_label = 'circle'
    for i in range(N):
        s = sample_neighbor(t, s_label, t_label, rad, w, h, sigma_s, sigma_p, sigma_th)
        draw_links(img1, s.x, s.y, s.w, s.h, s.theta, green, True);
    cv2.circle(img1, (int(s.x), int(s.y)), 4, [255, 0, 0], -1)
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title(title)

    # img2
    title = 'Given the inner link ' + str(node_id) + ' (red) the outer link samples for node ' + str(node_id + 4)
    t = Sample(gt_links[node_id-1, 0], gt_links[node_id-1, 1], node_id*np.pi/2 + gt_links[node_id-1, 2], w, h)
    s_label = 'link_' + str(node_id + 4)
    t_label = 'link_' + str(node_id)
    for i in range(N):
        s = sample_neighbor(t, s_label, t_label, rad, w, h, sigma_s, sigma_p, sigma_th)
        draw_links(img2, s.x, s.y, s.w, s.h, s.theta, green, True)
    cv2.circle(img2, (int(s.x), int(s.y)), 4, [255, 0, 0], -1)
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title(title)

    # img3
    title = 'Given the inner link ' + str(node_id) + ' (red) the circle samples'
    t = Sample(gt_links[node_id-1, 0], gt_links[node_id-1, 1], node_id*np.pi/2 + gt_links[node_id-1, 2], w, h)
    s_label = 'circle'
    t_label = 'link_' + str(node_id)
    for i in range(N):
        s = sample_neighbor(t, s_label, t_label, rad, w, h, sigma_s, sigma_p, sigma_th)
        draw_circle(img3, s.x, s.y, s.w);
    cv2.circle(img3, (int(s.x), int(s.y)), 4, [255, 0, 0], -1)
    axes[1, 0].imshow(img3)
    axes[1, 0].set_title(title)

    # img4
    title = 'Given the outer link ' + str(node_id + 4) + ' (red) the inner link samples for node ' + str(node_id)
    t = Sample(gt_links[node_id+4-1, 0], gt_links[node_id+4-1, 1], node_id*np.pi/2 + gt_links[node_id+4-1, 2], w, h)
    s_label = 'link_' + str(node_id)
    t_label = 'link_' + str(node_id + 4)
    for i in range(N):
        s = sample_neighbor(t, s_label, t_label, rad, w, h, sigma_s, sigma_p, sigma_th)
        draw_links(img4, s.x, s.y, s.w, s.h, s.theta, green, True)
    cv2.circle(img4, (int(s.x), int(s.y)), 4, [255, 0, 0], -1)
    axes[1, 1].imshow(img4)
    axes[1, 1].set_title(title)


if __name__=='__main__':
    test_pairwise_sampling(2)
    plt.show()
