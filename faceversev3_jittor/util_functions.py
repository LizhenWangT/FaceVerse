import numpy as np
import cv2


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def get_length(pred):
    #lm = np.array(pred)
    #brow_avg = (lm[19] + lm[24]) * 0.5
    #bottom = lm[8]
    #length = distance(brow_avg, bottom)
    center = pred[197]
    mouth_c = (pred[13] + pred[14]) / 2
    length = distance(mouth_c, center) * 4.2 / 2

    return length * 1.25


def ply_from_array(points, faces, output_file):

    num_points = len(points)
    num_triangles = len(faces)

    header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
element face {}
property list uchar int vertex_indices
end_header\n'''.format(num_points, num_triangles)

    with open(output_file,'w') as f:
        f.writelines(header)
        for item in points:
            f.write("{0:0.6f} {1:0.6f} {2:0.6f}\n".format(item[0], item[1], item[2]))

        for item in faces:
            number = len(item)
            row = "{0}".format(number)
            for elem in item:
                row += " {0} ".format(elem)
            row += "\n"
            f.write(row)


def ply_from_array_color(points, colors, faces, output_file):

    num_points = len(points)
    num_triangles = len(faces)

    header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {}
property list uchar int vertex_indices
end_header\n'''.format(num_points, num_triangles)

    with open(output_file,'w') as f:
        f.writelines(header)
        index = 0
        for item in points:
            f.write("{0:0.6f} {1:0.6f} {2:0.6f} {3} {4} {5}\n".format(item[0], item[1], item[2],
                                                        colors[index, 0], colors[index, 1], colors[index, 2]))
            index = index + 1

        for item in faces:
            number = len(item)
            row = "{0}".format(number)
            for elem in item:
                row += " {0} ".format(elem)
            row += "\n"
            f.write(row)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_pupil(img, lms, x_grid, y_grid, right=True):
    height, width, _ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(img_gray.shape, np.uint8)

    if right:
        ptsr = lms[36:42]#.reshape((-1, 1, 2))
    else:
        ptsr = lms[42:48]#.reshape((-1, 1, 2))
    mask_r = cv2.polylines(mask.copy(), [ptsr], True, 1)
    mask_r = cv2.fillPoly(mask_r, [ptsr], 1)

    img_eye_r = img_gray * mask_r + (1 - mask_r) * 255
    thres = int(np.min(img_eye_r)) + 30
    mask_r = mask_r.astype(np.float32) * (img_eye_r < thres).astype(np.float32)
    if np.sum(mask_r) < 10:
        return None, False
    r_eye_x = np.sum(x_grid * mask_r) / np.sum(mask_r)
    r_eye_y = np.sum(y_grid * mask_r) / np.sum(mask_r)
    
    pupil = np.array([r_eye_x, r_eye_y], dtype=np.float32)

    if right:
        center_eye_l = lms[36]
        center_eye_r = lms[39]
        center_eye_u = lms[37] / 2 + lms[38] / 2
        center_eye_d = lms[40] / 2 + lms[41] / 2
    else:
        center_eye_l = lms[42]
        center_eye_r = lms[45]
        center_eye_u = lms[43] / 2 + lms[44] / 2
        center_eye_d = lms[46] / 2 + lms[47] / 2
    center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4

    dis1_l = np.sqrt(np.sum(np.square(center_eye_l, center_eye_r)))
    dis2_l = np.sqrt(np.sum(np.square(center_eye_u, center_eye_d)))
    eye1_l = np.dot(pupil - center_eye, center_eye_r - center_eye_l) / dis1_l ** 2
    eye2_l = np.dot(pupil - center_eye, center_eye_d - center_eye_u) / dis2_l ** 2
    pupil = np.array([eye1_l, eye2_l], dtype=np.float32)
    return pupil, True


def draw_pupil(img, pred_lms, pupil_r, pupil_r_flag, pupil_l, pupil_l_flag):
    if pupil_r_flag:
        center_eye_l = pred_lms[36]
        center_eye_r = pred_lms[39]
        center_eye_u = pred_lms[37] / 2 + pred_lms[38] / 2
        center_eye_d = pred_lms[40] / 2 + pred_lms[41] / 2
        center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4
        pupil = center_eye + (center_eye_r - center_eye_l) * (pupil_r[0] + 0.0) + (center_eye_d - center_eye_u) * pupil_r[1]
        pupil = (pupil + 0.5).astype(np.int32)
        cv2.circle(img, (int(pupil[0]), int(pupil[1])), 3, [0, 255, 0], -1)
    if pupil_l_flag:
        center_eye_l = pred_lms[42]
        center_eye_r = pred_lms[45]
        center_eye_u = pred_lms[43] / 2 + pred_lms[44] / 2
        center_eye_d = pred_lms[46] / 2 + pred_lms[47] / 2
        center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4
        pupil = center_eye + (center_eye_r - center_eye_l) * (pupil_l[0] - 0.0) + (center_eye_d - center_eye_u) * pupil_l[1]
        pupil = (pupil + 0.5).astype(np.int32)
        cv2.circle(img, (int(pupil[0]), int(pupil[1])), 3, [0, 255, 0], -1)
    return img


def get_point_buf(ver, faces):
    point_buf=np.zeros((ver.shape[0], 8), np.int64)-1
    for i in range(faces.shape[0]):
        for j in range(8):
            if point_buf[faces[i, 0], j] < 0:
                point_buf[faces[i, 0], j] = i
                break
    for i in range(faces.shape[0]):
        for j in range(8):
            if point_buf[faces[i, 1], j] < 0:
                point_buf[faces[i, 1], j] = i
                break
    for i in range(faces.shape[0]):
        for j in range(8):
            if point_buf[faces[i, 2], j] < 0:
                point_buf[faces[i, 2], j] = i
                break
    for i in range(point_buf.shape[0]):
        num = 0
        for j in range(8):
            if point_buf[i, j] < 0:
                num += 1
        if num == 8:
            point_buf[i, :] *= 0
        if num == 7:
            point_buf[i, 1] = point_buf[i, 0]
            point_buf[i, 2] = point_buf[i, 0]
            point_buf[i, 3] = point_buf[i, 0]
            point_buf[i, 4] = point_buf[i, 0]
            point_buf[i, 5] = point_buf[i, 0]
            point_buf[i, 6] = point_buf[i, 0]
            point_buf[i, 7] = point_buf[i, 0]
        if num == 6:
            point_buf[i, 2] = point_buf[i, 0]
            point_buf[i, 3] = point_buf[i, 0]
            point_buf[i, 4] = point_buf[i, 0]
            point_buf[i, 5] = point_buf[i, 1]
            point_buf[i, 6] = point_buf[i, 1]
            point_buf[i, 7] = point_buf[i, 1]
        if num == 5:
            point_buf[i, 3] = point_buf[i, 0]
            point_buf[i, 4] = point_buf[i, 0]
            point_buf[i, 5] = point_buf[i, 1]
            point_buf[i, 6] = point_buf[i, 1]
            point_buf[i, 7] = point_buf[i, 2]
        if num == 4:
            point_buf[i, 4] = point_buf[i, 0]
            point_buf[i, 5] = point_buf[i, 1]
            point_buf[i, 6] = point_buf[i, 2]
            point_buf[i, 7] = point_buf[i, 3]
        elif num == 3:
            point_buf[i, 5] = point_buf[i, 1]
            point_buf[i, 6] = point_buf[i, 2]
            point_buf[i, 7] = point_buf[i, 3]
        elif num == 2:
            point_buf[i, 6] = point_buf[i, 2]
            point_buf[i, 7] = point_buf[i, 3]
        elif num == 1:
            point_buf[i, 7] = point_buf[i, 3]
    n = 0
    for i in range(point_buf.shape[0]):
        num = 0
        for j in range(6):
            if point_buf[i, j] < 0:
                num += 1
        if num > 0:
            n+=1
            print(num)
    print('point_buf errors:', n)
    return point_buf
