import numpy as np
import jittor as jt
jt.flags.use_cuda = 1


def photo_loss(pred_img, gt_img):
    loss = jt.sqrt(jt.sum((pred_img - gt_img).pow(2), 1))
    loss = jt.mean(loss)

    return loss

lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
l_eye = [263, 249, 390, 373, 374, 380, 381, 382, 263, 466, 388, 387, 386, 385, 384, 398]
l_brow = [276, 283, 282, 295, 300, 293, 334, 296]
r_eye = [33, 7, 163, 144, 145, 153, 154, 155, 33, 246, 161, 160, 159, 158, 157, 173]
r_brow = [46, 53, 52, 65, 70, 63, 105, 66]
oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def get_lm_weights():
    w = jt.ones(478)
    #w[0:17] = 1
    #w[28:31] = 5
    #w[36:48] = 5
    w[lips] = 5
    w[l_eye] = 5
    w[r_eye] = 5
    w[l_brow] = 5
    w[r_brow] = 5
    w[468:] = 2
    norm_w = w / w.sum()
    return norm_w


def lm_loss(pred_lms, gt_lms, weight, img_size=224):
    loss = jt.sum((pred_lms / img_size - gt_lms / img_size).pow(2), dim=2) * weight.reshape(1, -1)
    loss = jt.mean(loss.sum(1))

    return loss


def get_l2(tensor):
    return (tensor).pow(2).sum()


def reflectance_loss(tex, skin_mask):

    skin_mask = skin_mask.unsqueeze(2)
    tex_mean = jt.sum(tex*skin_mask, 1, keepdims=True)/jt.sum(skin_mask)
    loss = jt.sum(((tex-tex_mean) * skin_mask / 255.).pow(2)) / \
        (tex.shape[0] * jt.sum(skin_mask))

    return loss


def gamma_loss(gamma):

    gamma = gamma.reshape(-1, 3, 9)
    gamma_mean = jt.mean(gamma, dim=1, keepdims=True)
    gamma_loss = jt.mean((gamma - gamma_mean).pow(2))

    return gamma_loss
