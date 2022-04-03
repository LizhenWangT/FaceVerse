import cv2
import os
import numpy as np
import time
import torch
import argparse
from tqdm import tqdm

from model import get_faceverse
import model.losses as losses

from data_reader import ImageReader
from util_functions import get_length, ply_from_array_color

from network import Generator
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)


def init_optim_with_id(args, faceverse_model):
    rigid_optimizer = torch.optim.Adam([faceverse_model.get_rot_tensor(),
                                        faceverse_model.get_trans_tensor(),
                                        faceverse_model.get_id_tensor(),
                                        faceverse_model.get_exp_tensor()],
                                        lr=args.rf_lr)
    nonrigid_optimizer = torch.optim.Adam(
        [faceverse_model.get_id_tensor(), faceverse_model.get_exp_tensor(),
        faceverse_model.get_gamma_tensor(), faceverse_model.get_tex_tensor(),
        faceverse_model.get_rot_tensor(), faceverse_model.get_trans_tensor()], lr=args.nrf_lr)
    return rigid_optimizer, nonrigid_optimizer


def fit(args, device):
    faceverse_model, faceverse_dict = get_faceverse(version=args.version, batch_size=1, focal=1315, img_size=args.tar_size, device=device)
    lm_weights = losses.get_lm_weights(device)
    imagereader = ImageReader(args.input)

    uv_base = faceverse_dict['uv']
    meantex = faceverse_dict['meantex'].reshape(-1, 3)
    # normalize the texture
    bm, gm, rm = np.mean(meantex[:, 2]), np.mean(meantex[:, 1]), np.mean(meantex[:, 0])
    bs, gs, rs = np.std(meantex[:, 2]), np.std(meantex[:, 1]), np.std(meantex[:, 0])
    mean_tensor = torch.tensor([rm, gm, bm], dtype=torch.float32, requires_grad=False, device=device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std_tensor = torch.tensor([rs, gs, bs], dtype=torch.float32, requires_grad=False, device=device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    if args.version == 0:
        detail_path = 'data/faceverse_detail_v0.npy'
        detail_ckpt = 'data/faceverse_ckpt_detail_v0_100000.pt'
        exp_ckpt = 'data/faceverse_ckpt_exp_v0_150000.pt'
        noise_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
    elif args.version == 1:
        detail_path = 'data/faceverse_detail_v1.npy'
        detail_ckpt = 'data/faceverse_ckpt_detail_v1_124000.pt'
        exp_ckpt = 'data/faceverse_ckpt_exp_v1_140000.pt'
        noise_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    faceverse_detail_dict = np.load(detail_path, allow_pickle=True).item()

    g_detail = Generator(1024, 512, 8, 6, 6)
    g_detail = g_detail.to(device)
    g_detail.load_state_dict(torch.load(detail_ckpt)["g_ema"], strict=False)
    g_detail.eval()

    g_exp = Generator(1024, 512, 8, 3, 6)
    g_exp = g_exp.to(device)
    g_exp.load_state_dict(torch.load(exp_ckpt)["g_ema"], strict=False)
    g_exp.eval()

    uv_detail = torch.tensor(faceverse_detail_dict['uv'], dtype=torch.int64, requires_grad=False, device=device)
    mask_detail = torch.tensor(faceverse_detail_dict['uvmask'], dtype=torch.float32, requires_grad=False, device=device).unsqueeze(0).unsqueeze(0)
    tri_detail = torch.tensor(faceverse_detail_dict['tri'], dtype=torch.int64, requires_grad=False, device=device)
    point_buf_detail = torch.tensor(faceverse_detail_dict['point_buf'], dtype=torch.int64, requires_grad=False, device=device)
    keypoints_detail = torch.tensor(faceverse_detail_dict['keypoints']).squeeze().long().to(device)

    os.makedirs(args.res_folder, exist_ok=True)

    frame_ind = 0
    start_t = time.time()
    while True:
        # load data
        face_detected, frame, lms, frame_num = imagereader.get_data()
        if not face_detected:
            if frame:
                break
            else:
                continue
        
        imagename = imagereader.imagelist[frame_num - 1]
        basename = imagename.split('.')[0]
        print('Processing:', imagename)
        frame_ind += 1

        # init crop parameters and optimizer
        if args.align:
            border = 500
            half_length = int(get_length(lms))
            crop_center = lms[29].copy() + border
            frame_b = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
            align = cv2.resize(frame_b[crop_center[1] - half_length:crop_center[1] + half_length, crop_center[0] - half_length:crop_center[0] + half_length], 
                                (args.tar_size, args.tar_size), cv2.INTER_AREA)
            resized_lms = (lms - (crop_center - half_length - border)[np.newaxis, :]) / half_length / 2 * args.tar_size
            resized_lms = resized_lms.astype(np.int64)
            lms = torch.from_numpy(resized_lms[np.newaxis, :, :]).type(torch.float32).to(device)
            img_tensor = torch.from_numpy(align[np.newaxis, ...]).type(torch.float32).to(device)
        else:
            align = cv2.resize(frame, (args.tar_size, args.tar_size))
            lms[:, 0] = lms[:, 0] / frame.shape[1] * args.tar_size
            lms[:, 1] = lms[:, 1] / frame.shape[0] * args.tar_size
            lms = torch.from_numpy(lms[np.newaxis, :, :]).type(torch.float32).to(device)
            img_tensor = torch.from_numpy(align[np.newaxis, ...]).type(torch.float32).to(device)

        rigid_optimizer, nonrigid_optimizer = init_optim_with_id(args, faceverse_model)
        
        # fitting using only landmarks
        for i in range(args.rf_iters):
            rigid_optimizer.zero_grad()
            
            pred_dict = faceverse_model(faceverse_model.get_packed_tensors(), render=False, texture=False)
            lm_loss_val = losses.lm_loss(pred_dict['lms_proj'], lms, lm_weights, img_size=args.tar_size)
            exp_reg_loss = losses.get_l2(faceverse_model.get_exp_tensor())
            id_reg_loss = losses.get_l2(faceverse_model.get_id_tensor())
            total_loss = args.lm_loss_w * lm_loss_val + id_reg_loss*args.id_reg_w + exp_reg_loss*args.exp_reg_w
            
            total_loss.backward()
            rigid_optimizer.step()
        
        # fitting with differentiable rendering
        for i in range(args.nrf_iters):
            nonrigid_optimizer.zero_grad()

            pred_dict = faceverse_model(faceverse_model.get_packed_tensors(), render=True, texture=True)
            rendered_img = pred_dict['rendered_img']
            lms_proj = pred_dict['lms_proj']
            face_texture = pred_dict['face_texture']
            mask = rendered_img[:, :, :, 3].detach()

            lm_loss_val = losses.lm_loss(lms_proj, lms, lm_weights,img_size=args.tar_size)
            photo_loss_val = losses.photo_loss(rendered_img[:, :, :, :3], img_tensor, mask > 0)
            exp_reg_loss = losses.get_l2(faceverse_model.get_exp_tensor())
            id_reg_loss = losses.get_l2(faceverse_model.get_id_tensor())
            tex_reg_loss = losses.get_l2(faceverse_model.get_tex_tensor())
            tex_loss_val = losses.reflectance_loss(face_texture, faceverse_model.get_skinmask())

            loss = lm_loss_val*args.lm_loss_w + id_reg_loss*args.id_reg_w + exp_reg_loss*args.exp_reg_w + \
                    tex_reg_loss*args.tex_reg_w + tex_loss_val*args.tex_w + photo_loss_val*args.rgb_loss_w

            loss.backward()
            nonrigid_optimizer.step()
        
        # save data
        with torch.no_grad():
            pred_dict = faceverse_model(faceverse_model.get_packed_tensors(), render=True, texture=True)
            rendered_img_c = pred_dict['rendered_img']
            rendered_img_c = np.clip(rendered_img_c.cpu().squeeze().numpy(), 0, 255)
            pred_dict = faceverse_model(faceverse_model.get_packed_tensors(), render=True, texture=False)
            rendered_img_r = pred_dict['rendered_img']
            rendered_img_r = np.clip(rendered_img_r.cpu().squeeze().numpy(), 0, 255)
        mask_img_c = (rendered_img_c[:, :, 3:4] > 0).astype(np.uint8)
        drive_img_c = rendered_img_c[:, :, :3].astype(np.uint8) * mask_img_c + align * (1 - mask_img_c)
        mask_img_r = (rendered_img_r[:, :, 3:4] > 0).astype(np.uint8)
        drive_img_r = rendered_img_r[:, :, :3].astype(np.uint8) * mask_img_r + align * (1 - mask_img_r)
        drive_img = np.concatenate([align, drive_img_c, drive_img_r], axis=1)
        
        cv2.imwrite( os.path.join(args.res_folder, f'{basename}_base.png'), drive_img[:, :, ::-1])
        print(f'Speed:{(time.time() - start_t) / frame_ind:.4f}, {frame_ind:4} / {imagereader.num_frames:4}, {total_loss.item():.4f}')

        with torch.no_grad():
            id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = faceverse_model.split_coeffs(faceverse_model.get_packed_tensors())
            vertices = faceverse_model.get_vs(id_coeff, exp_coeff).cpu().numpy().squeeze()
            vertices_wo_exp = faceverse_model.get_vs(id_coeff, exp_coeff * 0).cpu().numpy().squeeze()
            colors = torch.clip(faceverse_model.get_color(tex_coeff), 0, 255).cpu().numpy().squeeze().astype(np.uint8)
            rotation = faceverse_model.compute_rotation_matrix(angles)

        if args.save_ply:
            output_ply = os.path.join(args.res_folder, f'{basename}_base.ply')
            ply_from_array_color(vertices, colors, faceverse_dict['tri'], output_ply)
        
        # fitting with the detail ckpt
        if args.version == 0:
            uv_tex = np.zeros((200, 200, 3), np.uint8)
            uv_geo = np.zeros((200, 200, 3), np.float32)
            uv_exp = np.zeros((200, 200, 3), np.float32)
            uv_tex[uv_base[:, 1], uv_base[:, 0]] = colors
            uv_geo[uv_base[:, 1], uv_base[:, 0]] = vertices_wo_exp
            uv_exp[uv_base[:, 1], uv_base[:, 0]] = vertices - vertices_wo_exp
        elif args.version == 1:
            uv_tex = np.zeros((256, 256, 3), np.uint8)
            uv_geo = np.zeros((256, 256, 3), np.float32)
            uv_exp = np.zeros((256, 256, 3), np.float32)
            uv_tex[uv_base[:, 1] + 28, uv_base[:, 0] + 28] = colors
            uv_geo[uv_base[:, 1] + 28, uv_base[:, 0] + 28] = vertices_wo_exp
            uv_exp[uv_base[:, 1] + 28, uv_base[:, 0] + 28] = vertices - vertices_wo_exp
        
        uv_tex = cv2.resize(uv_tex, (1024, 1024))
        uv_geo = cv2.resize(uv_geo, (1024, 1024))
        uv_exp = cv2.resize(uv_exp, (1024, 1024))
        bmt, gmt, rmt = np.mean(colors[:, 2]), np.mean(colors[:, 1]), np.mean(colors[:, 0])
        bst, gst, rst = np.std(colors[:, 2]), np.std(colors[:, 1]), np.std(colors[:, 0])
        uv_tex = torch.tensor(uv_tex, dtype=torch.float32, requires_grad=False, device=device).permute(2, 0, 1).unsqueeze(0)
        uv_geo = torch.tensor(uv_geo, dtype=torch.float32, requires_grad=False, device=device).permute(2, 0, 1).unsqueeze(0)
        uv_exp = torch.tensor(uv_exp, dtype=torch.float32, requires_grad=False, device=device).permute(2, 0, 1).unsqueeze(0)
        tmean_tensor = torch.tensor([rmt, gmt, bmt], dtype=torch.float32, requires_grad=False, device=device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        tstd_tensor = torch.tensor([rst, gst, bst], dtype=torch.float32, requires_grad=False, device=device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        uv_tex = torch.clip(((uv_tex - tmean_tensor) / tstd_tensor * std_tensor + mean_tensor) / 127.5 - 1, -1, 1)
        
        input_detail = torch.cat((uv_tex, uv_geo), dim=1)

        latent_detail = torch.randn(1000, 1, 512, device=device)
        latent_detail = torch.mean(latent_detail, dim=0)
        latent_detail.requires_grad = True

        noises_detail = [torch.randn(1, 1, 2 ** 2, 2 ** 2, requires_grad=True, device=device)]
        for i in range(3, g_detail.log_size + 1):
            for _ in range(2):
                noises_detail.append(torch.randn(1, 1, 2 ** i, 2 ** i, requires_grad=True, device=device))
        noises_grad_detail = []
        for num in noise_list:
            noise = noises_detail[num]
            noise.requires_grad = True
            noises_grad_detail.append(noise)
        
        detail_optimizer = torch.optim.Adam([latent_detail] + noises_grad_detail, lr=args.network_lr)
        pbar = tqdm(range(args.network_iters), initial=0, dynamic_ncols=True, smoothing=0.01)
        for i in pbar:
            detail_img, _ = g_detail([latent_detail], input_detail * mask_detail, return_latents=True, noise=noises_detail)
            detail_img_geo = detail_img[:, 3:] + uv_exp
            vs_detail = detail_img_geo[:, :, uv_detail[:, 1], uv_detail[:, 0]].permute(0, 2, 1)
            detail_img_tex = (detail_img[:, :3] + 1) * 127.5
            detail_img_tex = (detail_img_tex - mean_tensor) / std_tensor * tstd_tensor + tmean_tensor
            tx_detail = detail_img_tex[:, :, uv_detail[:, 1], uv_detail[:, 0]].permute(0, 2, 1)

            vs_t = faceverse_model.rigid_transform(vs_detail, rotation, translation)
            lms_t = vs_t[:, keypoints_detail, :]
            lms_proj = faceverse_model.project_vs(lms_t)
            lms_proj = torch.stack([lms_proj[:, :, 0], args.tar_size - lms_proj[:, :, 1]], dim=2)

            face_norm = faceverse_model.compute_norm(vs_detail, tri_detail, point_buf_detail)
            face_norm_r = face_norm.bmm(rotation)
            face_color = faceverse_model.add_illumination(tx_detail, face_norm_r, gamma)
            face_color_tv = TexturesVertex(face_color)

            mesh = Meshes(vs_t, tri_detail.repeat(1, 1, 1), face_color_tv)
            rendered_img = faceverse_model.renderer.alb_renderer(mesh)

            mask_t = rendered_img[:, :, :, 3].detach()
            photo_loss_val = losses.photo_loss(rendered_img[:, :, :, :3], img_tensor, mask_t > 0)
            lm_loss_val = losses.lm_loss(lms_proj, lms, lm_weights, img_size=args.tar_size)
            noise_loss_val = torch.square(noises_grad_detail[0]).mean()
            for noise in noises_grad_detail[1:]:
                noise_loss_val += torch.square(noise).mean()
            loss = lm_loss_val * 3e2 + photo_loss_val * 1.6 + noise_loss_val * 1e-4 #+ geo_reg_loss_val * 10
            pbar.set_description((f"lm: {lm_loss_val * 3e2:.4f}; pho: {photo_loss_val * 1.6:.4f}; "
                                f"noise: {noise_loss_val * 1e-4:.4f}; "))

            detail_optimizer.zero_grad()
            loss.backward()
            detail_optimizer.step()

        with torch.no_grad():
            detail_tex = np.clip(rendered_img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            detail_mask = (detail_tex[:, :, 3:4] > 0).astype(np.uint8)
            detail_tex = detail_tex[:, :, :3]
            detail_tex = align * (1 - detail_mask) + detail_tex * detail_mask

            face_color_tv = TexturesVertex(face_color * 0 + 200)
            mesh = Meshes(vs_t, tri_detail.repeat(1, 1, 1), face_color_tv)
            rendered_img = faceverse_model.renderer.sha_renderer(mesh)
            detail_render = np.clip(rendered_img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            detail_mask = (detail_render[:, :, 3:4] > 0).astype(np.uint8)
            detail_render = detail_render[:, :, :3]
            detail_render = align * (1 - detail_mask) + detail_render * detail_mask

        detail_img = np.concatenate([align, detail_tex, detail_render], axis=1)
        cv2.imwrite( os.path.join(args.res_folder, f'{basename}_detail.png'), detail_img[:, :, ::-1])

        if args.save_ply:
            vertices = vs_detail.detach().cpu().numpy().squeeze()
            colors = np.clip(tx_detail.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            output_ply = os.path.join(args.res_folder, f'{basename}_detail.ply')
            ply_from_array_color(vertices, colors, faceverse_detail_dict['tri'], output_ply)

        # fitting with the exp ckpt
        input_exp = torch.cat((detail_img_geo.detach().clone() * mask_detail + uv_exp, uv_exp), dim=1)

        latent_exp = torch.randn(1000, 1, 512, device=device)
        latent_exp = torch.mean(latent_exp, dim=0)
        latent_exp.requires_grad = True

        noises_exp = [torch.randn(1, 1, 2 ** 2, 2 ** 2, requires_grad=True, device=device)]
        for i in range(3, g_exp.log_size + 1):
            for _ in range(2):
                noises_exp.append(torch.randn(1, 1, 2 ** i, 2 ** i, requires_grad=True, device=device))
        noises_grad_exp = []
        for num in noise_list:
            noise = noises_exp[num]
            noise.requires_grad = True
            noises_grad_exp.append(noise)
        
        exp_optimizer = torch.optim.Adam([latent_exp] + noises_grad_exp, lr=args.network_lr)
        pbar = tqdm(range(args.network_iters), initial=0, dynamic_ncols=True, smoothing=0.01)
        for i in pbar:
            exp_img, _ = g_exp([latent_exp], input_exp * mask_detail, return_latents=True, noise=noises_exp)
            vs_exp = exp_img[:, :, uv_detail[:, 1], uv_detail[:, 0]].permute(0, 2, 1)

            vs_t = faceverse_model.rigid_transform(vs_exp, rotation, translation)
            lms_t = vs_t[:, keypoints_detail, :]
            lms_proj = faceverse_model.project_vs(lms_t)
            lms_proj = torch.stack([lms_proj[:, :, 0], args.tar_size - lms_proj[:, :, 1]], dim=2)

            face_norm = faceverse_model.compute_norm(vs_exp, tri_detail, point_buf_detail)
            face_norm_r = face_norm.bmm(rotation)
            face_color = faceverse_model.add_illumination(tx_detail.detach().clone(), face_norm_r, gamma)
            face_color_tv = TexturesVertex(face_color)

            mesh = Meshes(vs_t, tri_detail.repeat(1, 1, 1), face_color_tv)
            rendered_img = faceverse_model.renderer.alb_renderer(mesh)

            mask_t = rendered_img[:, :, :, 3].detach()
            photo_loss_val = losses.photo_loss(rendered_img[:, :, :, :3], img_tensor, mask_t > 0)
            lm_loss_val = losses.lm_loss(lms_proj, lms, lm_weights, img_size=args.tar_size)
            exp_reg_loss_val = torch.abs((uv_geo + uv_exp - exp_img) * mask_detail).mean()
            noise_loss_val = torch.square(noises_grad_exp[0]).mean()
            for noise in noises_grad_exp[1:]:
                noise_loss_val += torch.square(noise).mean()
            loss = lm_loss_val * 3e2 + photo_loss_val * 1.6 + noise_loss_val * 1e-4 + exp_reg_loss_val * 10
            pbar.set_description((f"lm: {lm_loss_val * 3e2:.4f}; pho: {photo_loss_val * 1.6:.4f}; "
                                f"noise: {noise_loss_val * 1e-4:.4f}; reg: {exp_reg_loss_val * 10:.4f}; "))

            exp_optimizer.zero_grad()
            loss.backward()
            exp_optimizer.step()

        with torch.no_grad():
            final_tex = np.clip(rendered_img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            final_mask = (final_tex[:, :, 3:4] > 0).astype(np.uint8)
            final_tex = final_tex[:, :, :3]
            final_tex = align * (1 - final_mask) + final_tex * final_mask

            face_color_tv = TexturesVertex(face_color * 0 + 200)
            mesh = Meshes(vs_t, tri_detail.repeat(1, 1, 1), face_color_tv)
            rendered_img = faceverse_model.renderer.sha_renderer(mesh)
            final_render = np.clip(rendered_img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            final_mask = (final_render[:, :, 3:4] > 0).astype(np.uint8)
            final_render = final_render[:, :, :3]
            final_render = align * (1 - final_mask) + final_render * final_mask

        final_img = np.concatenate([align, final_tex, final_render], axis=1)
        cv2.imwrite( os.path.join(args.res_folder, f'{basename}_final.png'), final_img[:, :, ::-1])

        if args.save_ply:
            vertices = vs_exp.detach().cpu().numpy().squeeze()
            output_ply = os.path.join(args.res_folder, f'{basename}_final.ply')
            ply_from_array_color(vertices, colors, faceverse_detail_dict['tri'], output_ply)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FaceVerse online tracker")

    parser.add_argument('--input', type=str, required=True,
                        help='input video path')
    parser.add_argument('--res_folder', type=str, required=True,
                        help='output directory')
    parser.add_argument('--save_ply', action="store_true",
                        help='save the output ply or not')
    parser.add_argument('--align', action="store_true",
                        help='align the input face')
    parser.add_argument('--version', type=int, default=1,
                        help='FaceVerse model version.')
    parser.add_argument('--tar_size', type=int, default=1024,
                        help='size for rendering window. We use a square window.')
    parser.add_argument('--padding_ratio', type=float, default=1.0,
                        help='enlarge the face detection bbox by a margin.')
    parser.add_argument('--faceverse_model', type=str, default='faceverse',
                        help='choose a 3dmm model, default: faceverse')
    parser.add_argument('--rf_iters', type=int, default=500,
                        help='iteration number of landmark fitting.')
    parser.add_argument('--nrf_iters', type=int, default=200,
                        help='iteration number of differentiable fitting.')
    parser.add_argument('--network_iters', type=int, default=800,
                        help='iteration number of network fitting.')
    parser.add_argument('--rf_lr', type=float, default=1e-2,
                        help='learning rate for landmark fitting')
    parser.add_argument('--nrf_lr', type=float, default=1e-2,
                        help='learning rate for differentiable fitting')
    parser.add_argument('--network_lr', type=float, default=2e-3,
                        help='learning rate for network fitting')
    parser.add_argument('--lm_loss_w', type=float, default=3e3,
                        help='weight for landmark loss')
    parser.add_argument('--rgb_loss_w', type=float, default=1.6,
                        help='weight for rgb loss')
    parser.add_argument('--id_reg_w', type=float, default=1e-3,
                        help='weight for id coefficient regularizer')
    parser.add_argument('--exp_reg_w', type=float, default=1.5e-4,
                        help='weight for expression coefficient regularizer')
    parser.add_argument('--tex_reg_w', type=float, default=3e-4,
                        help='weight for texture coefficient regularizer')
    parser.add_argument('--tex_w', type=float, default=1,
                        help='weight for texture reflectance loss.')

    args = parser.parse_args()

    device = 'cuda'
    
    fit(args, device)


