import jittor as jt
from jittor import Function
from faceverse_cuda.rasterizer_cuda import forward, backward, render


class Rasterizer(Function):
    def __init__(self, batch_size, image_size, ver_num):
        super(Rasterizer, self).__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.ver_num = ver_num
        
        self.render_colors = jt.zeros((self.batch_size, 4, self.image_size, self.image_size))
        self.grad_vertices = jt.zeros((self.batch_size, self.ver_num, 3))
        self.grad_textures = jt.zeros((self.batch_size, self.ver_num, 3))
        self.grad_normals = jt.zeros((self.batch_size, self.ver_num, 3))

    def execute(self, vertices, textures, normals):
        self.render_colors *= 0
        self.render_colors = forward(vertices, textures, normals, self.render_colors, self.image_size)[0]
        self.vertices = vertices
        self.textures = textures
        self.normals = normals
        return self.render_colors

    def grad(self, grad_colors):
        self.grad_vertices *= 0
        self.grad_textures *= 0
        self.grad_vertices, self.grad_textures = \
            backward(self.render_colors, grad_colors, self.vertices, self.textures, self.normals,
                    self.grad_vertices, self.grad_textures, self.image_size)
        return self.grad_vertices, self.grad_textures, self.grad_normals

    def render(self, face_vertices, face_textures):
        self.render_colors *= 0
        batch_size, num_faces = face_vertices.shape[:2]
        self.faces_info = jt.zeros((batch_size, num_faces, 9 * 3))
        _, self.render_colors = render(face_vertices, face_textures, self.faces_info, self.render_colors, self.image_size)
        return self.render_colors


