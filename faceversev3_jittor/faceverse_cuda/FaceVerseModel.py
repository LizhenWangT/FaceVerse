import numpy as np
import jittor as jt 
from jittor import nn 
jt.flags.use_cuda = 1

from faceverse_cuda.renderer import Renderer
from util_functions import get_point_buf


class FaceVerseModel(nn.Module):
    def __init__(self, model_dict, batch_size=1,
                 focal=1315, img_size=512):
        super(FaceVerseModel, self).__init__()

        self.focal = focal
        self.batch_size = batch_size
        self.img_size = img_size

        self.p_mat = self._get_p_mat()
        self.R = self._get_camera_R()
        self.camera_pos = self._get_camera_pose()
        self.rotXYZ = jt.init.eye([3, 3], dtype=jt.float32).view(1, 3, 3).repeat(3, 1, 1).view(3, 1, 3, 3)

        self.kp_inds = jt.array(model_dict['mediapipe_keypoints'].flatten(), dtype=jt.int64).stop_grad()
        self.ver_inds = model_dict['ver_inds']
        self.tri_inds = model_dict['tri_inds']

        meanshape = model_dict['meanshape'].reshape(-1, 3)
        meanshape[:, [1, 2]] *= -1
        meanshape = meanshape * 0.01
        
        self.meanshape = jt.array(meanshape.reshape(1, -1), dtype=jt.float32).stop_grad()
        self.meantex = jt.array(model_dict['meantex'].reshape(1, -1), dtype=jt.float32).stop_grad().repeat(self.batch_size, 1)

        idBase = model_dict['idBase'].reshape(-1, 150)
        exBase = model_dict['exBase_52'].reshape(-1, 3, 52)
        idBase[:, [1, 2]] *= -1
        idBase = idBase * 0.01
        exBase[:, [1, 2]] *= -1
        exBase = exBase * 0.01
        exBase = exBase.reshape(-1, 52)
        texBase = model_dict['texBase'].reshape(-1, 251)

        self.idBase = jt.array(idBase, dtype=jt.float32).stop_grad().unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.expBase = jt.array(exBase, dtype=jt.float32).stop_grad().unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.texBase = jt.array(texBase, dtype=jt.float32).stop_grad().unsqueeze(0).repeat(self.batch_size, 1, 1)

        self.tri = jt.array(model_dict['tri'], dtype=jt.float32).stop_grad()
        self.tri = self.tri[:, [0, 2, 1]]
        self.point_buf = jt.array(model_dict['point_buf'], dtype=jt.float32).stop_grad()
        self.face_mask = jt.array(model_dict['face_mask'], dtype=jt.float32).stop_grad()
        self.face_mask[int(self.ver_inds[2]):] += 1
        self.mask_num = int(jt.sum(self.face_mask).numpy()[0])
        self.face_mask = self.face_mask.unsqueeze(0).unsqueeze(2).repeat(self.batch_size, 1, 3)

        self.num_vertex = model_dict['meanshape'].shape[0]
        self.id_dims = self.idBase.shape[2]
        self.tex_dims = self.texBase.shape[2]
        self.exp_dims = self.expBase.shape[2]
        self.all_dims = self.id_dims + self.tex_dims + self.exp_dims

        self.init_coeff_tensors()

        self.renderer = Renderer(K=self.p_mat, R=self.R, t=self.camera_pos, image_size=self.img_size, 
                                 ver_num=int(self.ver_inds[2]), batch_size=self.batch_size)
        self.renderer_mask = Renderer(K=self.p_mat, R=self.R, t=self.camera_pos, image_size=self.img_size, 
                                 ver_num=self.mask_num, batch_size=self.batch_size)
        
        # for tracking by landmarks
        self.kp_inds_view = jt.concat([self.kp_inds[:, None] * 3, self.kp_inds[:, None] * 3 + 1, self.kp_inds[:, None] * 3 + 2], dim=1).flatten()
        self.idBase_view = self.idBase[:, self.kp_inds_view, :].detach().clone().stop_grad()
        self.expBase_view = self.expBase[:, self.kp_inds_view, :].detach().clone().stop_grad()
        self.meanshape_view = self.meanshape[:, self.kp_inds_view].detach().clone().stop_grad()

        # for rendering
        self.gamma_zeros = jt.zeros((self.batch_size, 27), dtype=jt.float32)
        self.gamma_zeros[:, [0, 9, 18]] += 0.1
        self.gamma_zeros[:, [1, 10, 19]] -= 0.2
        self.gamma_zeros[:, [2, 11, 20]] += 0.2
        self.gamma_zeros[:, [4, 13, 22]] -= 0.1

    def init_coeff_tensors(self):
        self.id_tensor = jt.zeros((1, self.id_dims), dtype=jt.float32)
        self.tex_tensor = jt.zeros((1, self.tex_dims), dtype=jt.float32)
        self.exp_tensor = jt.zeros((self.batch_size, self.exp_dims), dtype=jt.float32)
        self.gamma_tensor = jt.zeros((self.batch_size, 27), dtype=jt.float32)
        self.trans_tensor = jt.zeros((self.batch_size, 3), dtype=jt.float32)
        #self.trans_tensor[:, 2] -= 10 * self.focal / 1315
        self.eye_tensor = jt.zeros((self.batch_size, 4), dtype=jt.float32)
        self.rot_tensor = jt.zeros((self.batch_size, 3), dtype=jt.float32)

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def split_coeffs(self, coeffs):
        id_coeff = coeffs[:, :self.id_dims]  # identity(shape) coeff 
        exp_coeff = coeffs[:, self.id_dims:self.id_dims+self.exp_dims]  # expression coeff 
        tex_coeff = coeffs[:, self.id_dims+self.exp_dims:self.all_dims]  # texture(albedo) coeff 
        angles = coeffs[:, self.all_dims:self.all_dims+3] # ruler angles(x,y,z) for rotation of dim 3
        gamma = coeffs[:, self.all_dims+3:self.all_dims+30] # lighting coeff for 3 channel SH function of dim 27
        translation = coeffs[:, self.all_dims+30:self.all_dims+33]  # translation coeff of dim 3
        eye_coeff = coeffs[:, self.all_dims+33:]  # eye coeff of dim 3

        return id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff

    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff):
        coeffs = jt.concat([id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff], dim=1)
        return coeffs

    def get_packed_tensors(self):
        return self.merge_coeffs(self.id_tensor.repeat(self.batch_size, 1),
                                 self.exp_tensor,
                                 self.tex_tensor.repeat(self.batch_size, 1),
                                 self.rot_tensor, self.gamma_tensor,
                                 self.trans_tensor, self.eye_tensor)

    def execute(self, coeffs, render=False, surface=False, use_color=True):
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff = self.split_coeffs(coeffs)
        rotation = self.compute_rotation_matrix(angles)
        l_eye_mat = self.compute_eye_rotation_matrix(eye_coeff[:, :2])
        r_eye_mat = self.compute_eye_rotation_matrix(eye_coeff[:, 2:])
        l_eye_mean = self.get_l_eye_center(id_coeff)
        r_eye_mean = self.get_r_eye_center(id_coeff)

        if render:
            vs = self.get_vs(id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean)
            vs_t = self.rigid_transform(vs, rotation, translation)

            lms_t = self.get_lms(vs_t)
            lms_proj = self.project_vs(lms_t)
            lms_proj = jt.stack([lms_proj[:, :, 0], lms_proj[:, :, 1]], dim=2)
            colors = self.get_color(tex_coeff)
            norm = self.compute_norm(vs, self.tri, self.point_buf)
            norm_r = nn.bmm(norm, rotation)
            if use_color:
                colors_illumin = self.add_illumination(colors, norm_r, self.gamma_zeros)
            else:
                colors_illumin = self.add_illumination(colors * 0 + 130, norm_r, self.gamma_zeros)
            
            if surface:
                rendered_img = self.renderer.render(vs_t[:, :self.ver_inds[2]], colors_illumin[:, :self.ver_inds[2]], self.tri[:self.tri_inds[2]])
            else:
                rendered_img = self.renderer_mask(vs_t[self.face_mask>0].reshape(self.batch_size, self.mask_num, 3), colors_illumin[self.face_mask>0].reshape(self.batch_size, self.mask_num, 3), norm_r[self.face_mask>0].reshape(self.batch_size, self.mask_num, 3))
            
            return {'rendered_img': rendered_img,
                    'lms_proj': lms_proj,
                    'colors': colors,
                    'vertices': vs_t}
        else:
            lms = self.get_vs_lms(id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean)
            lms_t = self.rigid_transform(lms, rotation, translation)

            lms_proj = self.project_vs(lms_t)
            lms_proj = jt.stack([lms_proj[:, :, 0], lms_proj[:, :, 1]], dim=2)
            return {'lms_proj': lms_proj}

    def get_vs(self, id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean):
        face_shape = jt.matmul(self.idBase, id_coeff.unsqueeze(2)).squeeze(2) + \
            jt.matmul(self.expBase, exp_coeff.unsqueeze(2)).squeeze(2) + self.meanshape
        face_shape = face_shape.view(self.batch_size, -1, 3)
        face_shape[:, self.ver_inds[0]:self.ver_inds[1]] = jt.matmul(face_shape[:, self.ver_inds[0]:self.ver_inds[1]] - l_eye_mean, l_eye_mat) + l_eye_mean
        face_shape[:, self.ver_inds[1]:self.ver_inds[2]] = jt.matmul(face_shape[:, self.ver_inds[1]:self.ver_inds[2]] - r_eye_mean, r_eye_mat) + r_eye_mean
        return face_shape

    def get_vs_lms(self, id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean):
        face_shape = jt.matmul(self.idBase_view, id_coeff.unsqueeze(2)).squeeze(2) + \
            jt.matmul(self.expBase_view, exp_coeff.unsqueeze(2)).squeeze(2) + self.meanshape_view
        face_shape = face_shape.view(self.batch_size, -1, 3)
        face_shape[:, 473:478] = jt.matmul(face_shape[:, 473:478] - l_eye_mean, l_eye_mat) + l_eye_mean
        face_shape[:, 468:473] = jt.matmul(face_shape[:, 468:473] - r_eye_mean, r_eye_mat) + r_eye_mean
        return face_shape
    
    def get_l_eye_center(self, id_coeff):
        eye_shape = jt.matmul(self.idBase.reshape(self.batch_size, -1, 3, self.id_dims)[:, self.ver_inds[0]:self.ver_inds[1]].reshape(self.batch_size, -1, self.id_dims), 
                    id_coeff.unsqueeze(2)).squeeze(2) + self.meanshape.reshape(1, -1, 3)[:, self.ver_inds[0]:self.ver_inds[1]].reshape(1, -1)
        eye_shape = eye_shape.reshape(self.batch_size, -1, 3)
        eye_shape[:, :, 2] += 0.005
        return jt.mean(eye_shape, dim=1, keepdims=True).stop_grad()
    
    def get_r_eye_center(self, id_coeff):
        eye_shape = jt.matmul(self.idBase.reshape(self.batch_size, -1, 3, self.id_dims)[:, self.ver_inds[1]:self.ver_inds[2]].reshape(self.batch_size, -1, self.id_dims), 
                    id_coeff.unsqueeze(2)).squeeze(2) + self.meanshape.reshape(1, -1, 3)[:, self.ver_inds[1]:self.ver_inds[2]].reshape(1, -1)
        eye_shape = eye_shape.reshape(self.batch_size, -1, 3)
        eye_shape[:, :, 2] += 0.005
        return jt.mean(eye_shape, dim=1, keepdims=True).stop_grad()

    def get_color(self, tex_coeff):
        face_texture = jt.matmul(self.texBase, tex_coeff.unsqueeze(2)).squeeze(2) + self.meantex
        face_texture = face_texture.view(self.batch_size, -1, 3)
        return face_texture

    def _get_camera_pose(self):
        camera_pos = jt.float32([0.0, 0.0, 10.0]).reshape(1, 1, 3)
        return camera_pos.stop_grad()

    def _get_p_mat(self):
        half_image_width = self.img_size // 2
        p_matrix = np.array([self.focal, 0.0, half_image_width,
                             0.0, self.focal, half_image_width,
                             0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 3, 3)
        return jt.array(p_matrix, dtype=jt.float32).stop_grad()

    def _get_camera_R(self):
        R = np.reshape(np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], dtype=np.float32), [1, 3, 3])
        return jt.array(R, dtype=jt.float32).stop_grad()

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
        vs += self.camera_pos
        aug_projection = jt.matmul(vs, self.p_mat.repeat((self.batch_size, 1, 1)).permute((0, 2, 1)))
        face_projection = aug_projection[:, :, :2] / aug_projection[:, :, 2:3]
        return face_projection

    def compute_eye_rotation_matrix(self, eye):
        # 0 left_eye + down - up
        # 1 left_eye + right - left
        # 2 right_eye + down - up
        # 3 right_eye + right - left
        sinx = jt.sin(eye[:, 0])
        siny = jt.sin(eye[:, 1])
        cosx = jt.cos(eye[:, 0])
        cosy = jt.cos(eye[:, 1])
        if self.batch_size != 1:
            rotXYZ = self.rotXYZ.repeat(1, self.batch_size, 1, 1).detach().clone()
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

        rotation = nn.bmm(rotXYZ[1], rotXYZ[0])

        return rotation.permute(0, 2, 1)

    def compute_rotation_matrix(self, angles):
        sinx = jt.sin(angles[:, 0])
        siny = jt.sin(angles[:, 1])
        sinz = jt.sin(angles[:, 2])
        cosx = jt.cos(angles[:, 0])
        cosy = jt.cos(angles[:, 1])
        cosz = jt.cos(angles[:, 2])
        if self.batch_size != 1:
            rotXYZ = self.rotXYZ.repeat(1, self.batch_size, 1, 1).detach().clone()
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

        rotation = nn.bmm(rotXYZ[2], nn.bmm(rotXYZ[1], rotXYZ[0]))

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

        H = jt.stack(arrH, 1)
        Y = H.view(self.batch_size, face_texture.shape[1], 9)
        lighting = nn.bmm(Y, gamma)

        face_color = face_texture * lighting
        return face_color

    def rigid_transform(self, vs, rot, trans):
        vs_r = jt.matmul(vs, rot)
        vs_t = vs_r + trans.view(-1, 1, 3)
        return vs_t

