import torch
from torch import nn
import numpy as np

from model.ModelRenderer import ModelRenderer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

class FaceVerseModel(nn.Module):
    def __init__(self, model_dict, batch_size=1,
                 focal=1315, img_size=256, use_simplification=False, device='cuda:0'):
        super(FaceVerseModel, self).__init__()

        self.focal = focal
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = torch.device(device)

        self.p_mat = self._get_p_mat(device)
        self.reverse_z = self._get_reverse_z(device)
        self.camera_pos = self._get_camera_pose(device)
        self.rotXYZ = torch.eye(3).view(1, 3, 3).repeat(3, 1, 1).view(3, 1, 3, 3).to(self.device)

        self.renderer = ModelRenderer(self.focal, self.img_size, self.device)
        
        if use_simplification:
            self.select_id = model_dict['select_id']
            self.select_id_tris = np.vstack((self.select_id * 3, self.select_id * 3 + 1, self.select_id * 3 + 2)).transpose().flatten()
            self.skinmask = torch.tensor(model_dict['skinmask_select'], requires_grad=False, device=self.device)

            self.kp_inds = torch.tensor(model_dict['keypoints_select'].reshape(-1, 1), requires_grad=False).squeeze().long().to(self.device)

            self.meanshape = torch.tensor(model_dict['meanshape'].reshape(1, -1)[:, self.select_id_tris], dtype=torch.float32, requires_grad=False, device=self.device)
            self.meantex = torch.tensor(model_dict['meantex'].reshape(1, -1)[:, self.select_id_tris], dtype=torch.float32, requires_grad=False, device=self.device)

            self.idBase = torch.tensor(model_dict['idBase'][self.select_id_tris], dtype=torch.float32, requires_grad=False, device=self.device)
            self.expBase = torch.tensor(model_dict['exBase'][self.select_id_tris], dtype=torch.float32, requires_grad=False, device=self.device)
            self.texBase = torch.tensor(model_dict['texBase'][self.select_id_tris], dtype=torch.float32, requires_grad=False, device=self.device)

            self.tri = torch.tensor(model_dict['tri_select'], dtype=torch.int64, requires_grad=False, device=self.device)
            self.point_buf = torch.tensor(model_dict['point_buf_select'], dtype=torch.int64, requires_grad=False, device=self.device)
        
        else:
            self.skinmask = torch.tensor(model_dict['skinmask'], requires_grad=False, device=self.device)

            self.kp_inds = torch.tensor(model_dict['keypoints'].reshape(-1, 1), requires_grad=False).squeeze().long().to(self.device)

            self.meanshape = torch.tensor(model_dict['meanshape'].reshape(1, -1), dtype=torch.float32, requires_grad=False, device=self.device)
            self.meantex = torch.tensor(model_dict['meantex'].reshape(1, -1), dtype=torch.float32, requires_grad=False, device=self.device)

            self.idBase = torch.tensor(model_dict['idBase'], dtype=torch.float32, requires_grad=False, device=self.device)
            self.expBase = torch.tensor(model_dict['exBase'], dtype=torch.float32, requires_grad=False, device=self.device)
            self.texBase = torch.tensor(model_dict['texBase'], dtype=torch.float32, requires_grad=False, device=self.device)

            self.tri = torch.tensor(model_dict['tri'], dtype=torch.int64, requires_grad=False, device=self.device)
            self.point_buf = torch.tensor(model_dict['point_buf'], dtype=torch.int64, requires_grad=False, device=self.device)

        self.num_vertex = self.meanshape.shape[1] // 3
        self.id_dims = self.idBase.shape[1]
        self.tex_dims = self.texBase.shape[1]
        self.exp_dims = self.expBase.shape[1]
        self.all_dims = self.id_dims + self.tex_dims + self.exp_dims

        self.init_coeff_tensors()
        
        # for tracking by landmarks
        self.kp_inds_view = torch.cat([self.kp_inds[:, None] * 3, self.kp_inds[:, None] * 3 + 1, self.kp_inds[:, None] * 3 + 2], dim=1).flatten()
        self.idBase_view = self.idBase[self.kp_inds_view, :].detach().clone()
        self.expBase_view = self.expBase[self.kp_inds_view, :].detach().clone()
        self.meanshape_view = self.meanshape[:, self.kp_inds_view].detach().clone()

    def init_coeff_tensors(self):
        self.id_tensor = torch.zeros(
            (self.batch_size, self.id_dims), dtype=torch.float32,
            requires_grad=True, device=self.device)

        self.tex_tensor = torch.zeros(
            (self.batch_size, self.tex_dims), dtype=torch.float32,
            requires_grad=True, device=self.device)

        self.exp_tensor = torch.zeros(
            (self.batch_size, self.exp_dims), dtype=torch.float32,
            requires_grad=True, device=self.device)
    
        self.gamma_tensor = torch.zeros(
            (self.batch_size, 27), dtype=torch.float32,
            requires_grad=True, device=self.device)
    
        self.trans_tensor = torch.zeros(
            (self.batch_size, 3), dtype=torch.float32,
            requires_grad=False, device=self.device)
        self.trans_tensor[:, 2] += 6
        self.trans_tensor.requires_grad = True
    
        self.rot_tensor = torch.zeros(
            (self.batch_size, 3), dtype=torch.float32,
            requires_grad=False, device=self.device)
        self.rot_tensor[:, 0] += torch.pi
        self.rot_tensor.requires_grad = True

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def split_coeffs(self, coeffs):
        id_coeff = coeffs[:, :self.id_dims]  # identity(shape) coeff 
        exp_coeff = coeffs[:, self.id_dims:self.id_dims+self.exp_dims]  # expression coeff 
        tex_coeff = coeffs[:, self.id_dims+self.exp_dims:self.all_dims]  # texture(albedo) coeff 
        angles = coeffs[:, self.all_dims:self.all_dims+3] # ruler angles(x,y,z) for rotation of dim 3
        gamma = coeffs[:, self.all_dims+3:self.all_dims+30] # lighting coeff for 3 channel SH function of dim 27
        translation = coeffs[:, self.all_dims+30:]  # translation coeff of dim 3

        return id_coeff, exp_coeff, tex_coeff, angles, gamma, translation

    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, angles, gamma, translation):
        coeffs = torch.cat([id_coeff, exp_coeff, tex_coeff, angles, gamma, translation], dim=1)
        return coeffs

    def get_packed_tensors(self):
        return self.merge_coeffs(self.id_tensor.repeat(self.batch_size, 1),
                                 self.exp_tensor,
                                 self.tex_tensor.repeat(self.batch_size, 1),
                                 self.rot_tensor, self.gamma_tensor,
                                 self.trans_tensor)

    def forward(self, coeffs, render=True, texture=True):
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = self.split_coeffs(coeffs)
        rotation = self.compute_rotation_matrix(angles)

        if render:
            vs = self.get_vs(id_coeff, exp_coeff)
            vs_t = self.rigid_transform(vs, rotation, translation)

            lms_t = self.get_lms(vs_t)
            lms_proj = self.project_vs(lms_t)
            lms_proj = torch.stack(
                [lms_proj[:, :, 0], self.img_size-lms_proj[:, :, 1]], dim=2)
            face_texture = self.get_color(tex_coeff)
            face_norm = self.compute_norm(vs, self.tri, self.point_buf)
            face_norm_r = face_norm.bmm(rotation)
            face_color = self.add_illumination(face_texture, face_norm_r, gamma)

            if texture:
                face_color_tv = TexturesVertex(face_color)
                mesh = Meshes(vs_t, self.tri.repeat(self.batch_size, 1, 1), face_color_tv)
                rendered_img = self.renderer.alb_renderer(mesh)
            else:
                face_color_tv = TexturesVertex(face_color * 0 + 200)
                mesh = Meshes(vs_t, self.tri.repeat(self.batch_size, 1, 1), face_color_tv)
                rendered_img = self.renderer.sha_renderer(mesh)

            return {'rendered_img': rendered_img,
                    'lms_proj': lms_proj,
                    'face_texture': face_texture,
                    'vs': vs_t,
                    'tri': self.tri,
                    'color': face_color}
        else:
            lms = self.get_vs_lms(id_coeff, exp_coeff)
            lms_t = self.rigid_transform(
                lms, rotation, translation)

            lms_proj = self.project_vs(lms_t)
            lms_proj = torch.stack(
                [lms_proj[:, :, 0], self.img_size-lms_proj[:, :, 1]], dim=2)
            return {'lms_proj': lms_proj}

    def get_vs(self, id_coeff, exp_coeff):
        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
            torch.einsum('ij,aj->ai', self.expBase, exp_coeff) + self.meanshape
        face_shape = face_shape.view(self.batch_size, -1, 3)
        return face_shape

    def get_vs_lms(self, id_coeff, exp_coeff):
        face_shape = torch.einsum('ij,aj->ai', self.idBase_view, id_coeff) + \
            torch.einsum('ij,aj->ai', self.expBase_view, torch.abs(exp_coeff)) + self.meanshape_view
        face_shape = face_shape.view(self.batch_size, -1, 3)
        return face_shape

    def get_color(self, tex_coeff):
        face_texture = torch.einsum('ij,aj->ai', self.texBase, tex_coeff) + self.meantex
        face_texture = face_texture.view(self.batch_size, -1, 3)
        return face_texture

    def get_skinmask(self):
        return self.skinmask

    def _get_camera_pose(self, device):
        camera_pos = torch.tensor([0.0, 0.0, 10.0], device=device).reshape(1, 1, 3)
        return camera_pos

    def _get_p_mat(self, device):
        half_image_width = self.img_size // 2
        p_matrix = np.array([self.focal, 0.0, half_image_width,
                             0.0, self.focal, half_image_width,
                             0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 3, 3)
        return torch.tensor(p_matrix, device=device)

    def _get_reverse_z(self, device):
        reverse_z = np.reshape(np.array([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0], dtype=np.float32), [1, 3, 3])
        return torch.tensor(reverse_z, device=device)

    def compute_norm(self, vs, tri, point_buf):
        face_id = tri
        point_id = point_buf
        v1 = vs[:, face_id[:, 0], :]
        v2 = vs[:, face_id[:, 1], :]
        v3 = vs[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)

        v_norm = face_norm[:, point_id, :].sum(2)
        v_norm = v_norm / (v_norm.norm(dim=2).unsqueeze(2) + 1e-9)

        return v_norm

    def project_vs(self, vs):
        vs = torch.matmul(vs, self.reverse_z.repeat((self.batch_size, 1, 1))) + self.camera_pos
        aug_projection = torch.matmul(vs, self.p_mat.repeat((self.batch_size, 1, 1)).permute((0, 2, 1)))
        face_projection = aug_projection[:, :, :2] / torch.reshape(aug_projection[:, :, 2], [self.batch_size, -1, 1])
        return face_projection

    def compute_rotation_matrix(self, angles):
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        if self.batch_size != 1:
            rotXYZ = self.rotXYZ.repeat(1, self.batch_size * 3, 1, 1)
        else:
            rotXYZ = self.rotXYZ.detach().clone()

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)

    def add_illumination(self, face_texture, norm, gamma):
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8
        gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(a0 * c0 * (nx * 0 + 1))
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(self.batch_size, face_texture.shape[1], 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        return face_color

    def rigid_transform(self, vs, rot, trans):
        vs_r = torch.matmul(vs, rot)
        vs_t = vs_r + trans.view(-1, 1, 3)
        return vs_t

    def get_rot_tensor(self):
        return self.rot_tensor

    def get_trans_tensor(self):
        return self.trans_tensor

    def get_exp_tensor(self):
        return self.exp_tensor

    def get_tex_tensor(self):
        return self.tex_tensor

    def get_id_tensor(self):
        return self.id_tensor

    def get_gamma_tensor(self):
        return self.gamma_tensor

