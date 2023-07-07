import math
import jittor as jt
from jittor import nn
import numpy

from faceverse_cuda.transform import Projection
from faceverse_cuda.rasterizer import Rasterizer


class Renderer(nn.Module):
    def __init__(self, K, R, t, image_size, batch_size, ver_num):
        super(Renderer, self).__init__()
        # camera
        self.batch_size = batch_size
        self.transform = Projection(self.batch_size, K, R, t, image_size, ver_num)
        self.rasterizer = Rasterizer(self.batch_size, image_size, ver_num)

    def execute(self, vertices, textures, normals):
        vertices = self.transform(vertices)
        image = self.rasterizer(vertices, textures, normals)
        return image

    def render(self, vertices, textures, faces):
        vertices = self.transform(vertices)
        image = self.rasterizer.render(vertices[:, faces], textures[:, faces])
        return image
