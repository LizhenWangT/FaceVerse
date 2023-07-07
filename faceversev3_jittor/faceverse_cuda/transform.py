import jittor as jt
from jittor import nn


class Projection(nn.Module):
    def __init__(self, batch_size, K, R, t, image_size, ver_num):
        super(Projection, self).__init__()
        '''
        Calculate projective transformation of vertices given a projection matrix
        Input parameters:
        K: 1 * 3 * 3 intrinsic camera matrix
        R, t: 1 * 3 * 3, 1 * 1 * 3 extrinsic calibration parameters
        image_size: original size of image captured by the camera
        Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
        pixels and z is the depth
        '''
        self.K = K
        self.R = R
        self.t = t
        self.image_size = image_size
        self.z_ = jt.ones([batch_size, ver_num]).stop_grad()

    def execute(self, vertices, eps=1e-9):

        # instead of P*x we compute x'*P'
        vertices = jt.matmul(vertices, self.R.transpose((0,2,1))[0]) + self.t
        x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
        x_ = x / (z + eps)
        y_ = y / (z + eps)

        vertices = jt.stack([x_, y_, self.z_], dim=-1)
        vertices = jt.matmul(vertices, self.K.transpose((0,2,1))[0])
        u, v = vertices[:, :, 0], vertices[:, :, 1]
        v = self.image_size - v
        # map u,v from [0, img_size] to [-1, 1] to use by the renderer
        u = 2 * (u - self.image_size / 2.) / self.image_size
        v = 2 * (v - self.image_size / 2.) / self.image_size
        vertices = jt.stack([u, v, z], dim=-1)

        return vertices


