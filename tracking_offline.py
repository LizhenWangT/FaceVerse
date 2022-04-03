import cv2
import os
import numpy as np
import time
import torch
import argparse

from model import get_faceverse
import model.losses as losses

from data_reader import OfflineReader
from util_functions import get_length, ply_from_array_color


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


def tracking(args, device):
    faceverse_model, faceverse_dict = get_faceverse(version=args.version, batch_size=1, focal=1315, img_size=args.tar_size, device=device)
    lm_weights = losses.get_lm_weights(device)
    offreader = OfflineReader(args.input)
    print(args.input, 'FPS:', offreader.fps)

    os.makedirs(args.res_folder, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(args.res_folder, 'faceverse_tracking.mp4'), fourcc, offreader.fps, (args.tar_size * 3, args.tar_size))

    frame_ind = 0
    while True:
        # load data
        face_detected, frame, lms, frame_num = offreader.get_data()
        if not face_detected:
            if frame:
                out_video.release()
                exit()
            else:
                continue
        
        # init crop parameters and optimizer
        if frame_ind == 0:
            border = 500
            half_length = int(get_length(lms))
            crop_center = lms[29].copy() + border
            print('First frame:', half_length, crop_center)
            rigid_optimizer, nonrigid_optimizer = init_optim_with_id(args, faceverse_model)
        frame_b = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
        align = cv2.resize(frame_b[crop_center[1] - half_length:crop_center[1] + half_length, crop_center[0] - half_length:crop_center[0] + half_length], 
                            (args.tar_size, args.tar_size), cv2.INTER_AREA)
        resized_lms = (lms - (crop_center - half_length - border)[np.newaxis, :]) / half_length / 2 * args.tar_size
        resized_lms = resized_lms.astype(np.int64)
        
        lms = torch.from_numpy(resized_lms[np.newaxis, :, :]).type(torch.float32).to(device)
        img_tensor = torch.from_numpy(align[np.newaxis, ...]).type(torch.float32).to(device)

        if frame_ind == 0:
            num_iters_rf = args.first_rf_iters
            num_iters_nrf = args.first_nrf_iters
        else:
            num_iters_rf = args.rest_rf_iters
            num_iters_nrf = args.rest_nrf_iters
        
        # fitting using only landmarks
        for i in range(num_iters_rf):
            rigid_optimizer.zero_grad()
            
            pred_dict = faceverse_model(faceverse_model.get_packed_tensors(), render=False, texture=False)
            lm_loss_val = losses.lm_loss(pred_dict['lms_proj'], lms, lm_weights, img_size=args.tar_size)
            exp_reg_loss = losses.get_l2(faceverse_model.get_exp_tensor())
            id_reg_loss = losses.get_l2(faceverse_model.get_id_tensor())
            total_loss = args.lm_loss_w * lm_loss_val + id_reg_loss*args.id_reg_w + exp_reg_loss*args.exp_reg_w
            
            total_loss.backward()
            rigid_optimizer.step()
        
        # fitting with differentiable rendering
        for i in range(num_iters_nrf):
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
        if frame_ind == 0:
            start_t = time.time()
        frame_ind += 1
        
        out_video.write(drive_img[:, :, ::-1])
        #cv2.imwrite( os.path.join(args.res_folder, f'{str(frame_ind).zfill(4)}.png'), drive_img[:, :, ::-1])
        print(f'Speed:{(time.time() - start_t) / frame_ind:.4f}, {frame_ind:4} / {offreader.num_frames:4}, {total_loss.item():.4f}')

        if args.save_ply:
            vertices = pred_dict['vs'].detach().cpu().squeeze().numpy()
            colors = pred_dict['face_texture'].detach().cpu().squeeze().numpy()
            colors = np.clip(colors, 0, 255).astype(np.uint8)
            output_ply = os.path.join(args.res_folder, f'{str(frame_ind).zfill(4)}.ply')
            ply_from_array_color(vertices, colors, faceverse_dict['tri'], output_ply)
        
        if args.save_coeff:
            coeffs = faceverse_model.get_packed_tensors().detach().clone().cpu().numpy()
            np.save(os.path.join(args.res_folder, f'{str(frame_ind).zfill(4)}.npy'), coeffs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FaceVerse online tracker")

    parser.add_argument('--input', type=str, required=True,
                        help='input video path')
    parser.add_argument('--res_folder', type=str, required=True,
                        help='output directory')
    parser.add_argument('--save_ply', action="store_true",
                        help='save the output ply or not')
    parser.add_argument('--save_coeff', action="store_true",
                        help='save the output coeff or not')
    parser.add_argument('--version', type=int, default=1,
                        help='FaceVerse model version.')
    parser.add_argument('--tar_size', type=int, default=512,
                        help='size for rendering window. We use a square window.')
    parser.add_argument('--padding_ratio', type=float, default=1.0,
                        help='enlarge the face detection bbox by a margin.')
    parser.add_argument('--recon_model', type=str, default='faceverse',
                        help='choose a 3dmm model, default: faceverse')
    parser.add_argument('--first_rf_iters', type=int, default=500,
                        help='iteration number of landmark fitting for the first frame in video fitting.')
    parser.add_argument('--first_nrf_iters', type=int, default=300,
                        help='iteration number of differentiable fitting for the first frame in video fitting.')
    parser.add_argument('--rest_rf_iters', type=int, default=50,
                        help='iteration number of landmark fitting for the remaining frames in video fitting.')
    parser.add_argument('--rest_nrf_iters', type=int, default=30,
                        help='iteration number of differentiable fitting for the remaining frames in video fitting.')
    parser.add_argument('--rf_lr', type=float, default=1e-2,
                        help='learning rate for landmark fitting')
    parser.add_argument('--nrf_lr', type=float, default=1e-2,
                        help='learning rate for differentiable fitting')
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
    
    tracking(args, device)


