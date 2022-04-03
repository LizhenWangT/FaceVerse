import numpy as np


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def get_length(pred):
    lm = np.array(pred)
    brow_avg = (lm[19] + lm[24]) * 0.5
    bottom = lm[8]
    length = distance(brow_avg, bottom)

    return length * 1.05


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

