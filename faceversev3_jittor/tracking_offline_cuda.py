import cv2
import os
import numpy as np
import time
import jittor as jt
jt.flags.use_cuda = 1
import argparse
import threading
from queue import Queue

from faceverse_cuda import get_faceverse
import faceverse_cuda.losses as losses

from data_reader import OfflineReader
from util_functions import get_length, ply_from_array_color


num_queue = Queue()
out_queue = Queue()
image_queue = Queue()
back_queue = Queue()


class Tracking(threading.Thread):
    def __init__(self, args):
        super(Tracking, self).__init__()
        self.args = args
        self.fvm, self.fvd = get_faceverse(batch_size=self.args.batch_size, focal=int(1315 / 512 * self.args.tar_size), img_size=self.args.tar_size)
        self.lm_weights = losses.get_lm_weights()
        self.offreader = OfflineReader(args.input, args.back, args.tar_size, args.image_size, args.crop_size, skip_frames=args.skip_frames)
        self.thread_lock = threading.Lock()
        self.frame_ind = 0
        self.thread_exit = False
        self.queue_num = 0
        self.scale = 0
    
    def eyes_refine(self, eye_coeffs):
        for i in range(self.args.batch_size):
            for j in range(2):
                if eye_coeffs[i, j] > 0.4:
                    eye_coeffs[i, j] = (eye_coeffs[i, j] - 0.4) * 2 + 0.4
        return eye_coeffs

    def run(self):
        while not self.thread_exit:
            # load data
            detected, align, lms_detect, outimg, backimg, frame_num = self.offreader.get_data()
            if not detected:
                if not align:
                    continue
                else:
                    break
            lms = jt.array(lms_detect[None, :, :], dtype=jt.float32).stop_grad()
            img_tensor = jt.array(align[None, :, :, :], dtype=jt.float32).stop_grad().transpose((0, 3, 1, 2))

            if self.frame_ind == 0:
                num_iters_rf = 1000
                num_iters_nrf = 200
                rigid_optimizer = jt.optim.Adam([self.fvm.rot_tensor, self.fvm.trans_tensor, self.fvm.id_tensor, self.fvm.exp_tensor, self.fvm.eye_tensor], 
                                                    lr=1e-2, betas=(0.8, 0.95))
                nonrigid_optimizer = jt.optim.Adam([self.fvm.id_tensor, self.fvm.gamma_tensor, self.fvm.tex_tensor,
                                                    self.fvm.rot_tensor, self.fvm.trans_tensor, self.fvm.eye_tensor], lr=5e-3, betas=(0.5, 0.9))
            else:
                rigid_optimizer = jt.optim.Adam([self.fvm.rot_tensor, self.fvm.trans_tensor, self.fvm.exp_tensor, self.fvm.eye_tensor], 
                                                    lr=1e-3, betas=(0.5, 0.9))
                nonrigid_optimizer = jt.optim.Adam([self.fvm.exp_tensor, self.fvm.gamma_tensor, 
                                                    self.fvm.rot_tensor, self.fvm.trans_tensor, self.fvm.eye_tensor], lr=1e-3, betas=(0.5, 0.9))
                num_iters_rf = 100
                num_iters_nrf = 30

            scale = ((lms_detect - lms_detect.mean(0)) ** 2).mean() ** 0.5
            if self.scale != 0:
                self.fvm.trans_tensor[0, 2] = (self.fvm.trans_tensor[0, 2] + self.fvm.camera_pos[0, 0, 2]) * self.scale / scale - self.fvm.camera_pos[0, 0, 2]
                lms_center = jt.mean(lms, dim=1)
                self.fvm.trans_tensor[:, :2] -= (lms_center - lms_proj_center) * self.fvm.trans_tensor[:, 2:3] / self.fvm.focal * 0.5
            self.scale = scale

            # fitting using only landmarks (rigid)
            for i in range(num_iters_rf):
                pred_dict = self.fvm(self.fvm.get_packed_tensors(), render=False)
                lm_loss_val = losses.lm_loss(pred_dict['lms_proj'], lms, self.lm_weights, img_size=self.args.tar_size)
                exp_reg_loss = losses.get_l2(self.fvm.exp_tensor[:, 40:]) + losses.get_l2(self.fvm.exp_tensor[:, :40])
                if self.frame_ind == 0:
                    id_reg_loss = losses.get_l2(self.fvm.id_tensor)
                    loss = lm_loss_val * self.args.lm_loss_w + id_reg_loss * self.args.id_reg_w + exp_reg_loss * self.args.exp_reg_w
                else:
                    rt_reg_loss = losses.get_l2(self.fvm.rot_tensor - rot_c) + losses.get_l2(self.fvm.trans_tensor - trans_c)
                    loss = lm_loss_val * self.args.lm_loss_w + id_reg_loss * self.args.id_reg_w + \
                                 exp_reg_loss * self.args.exp_reg_w + rt_reg_loss * self.args.rt_reg_w
                rigid_optimizer.zero_grad()
                rigid_optimizer.backward(loss)
                rigid_optimizer.step()
                self.fvm.exp_tensor[self.fvm.exp_tensor < 0] *= 0
            
            if self.args.use_dr:
                # fitting with differentiable rendering
                for i in range(num_iters_nrf):
                    pred_dict = self.fvm(self.fvm.get_packed_tensors(), render=True)
                    rendered_img = pred_dict['rendered_img']
                    lms_proj = pred_dict['lms_proj']
                    face_texture = pred_dict['colors']
                    lm_loss_val = losses.lm_loss(lms_proj, lms, self.lm_weights,img_size=self.args.tar_size)
                    photo_loss_val = losses.photo_loss(rendered_img[:, :3], img_tensor)
                    exp_reg_loss = losses.get_l2(self.fvm.exp_tensor)
                    if self.frame_ind == 0:
                        id_reg_loss = losses.get_l2(self.fvm.id_tensor)
                        tex_reg_loss = losses.get_l2(self.fvm.tex_tensor)
                        loss = lm_loss_val * self.args.lm_loss_w + id_reg_loss * self.args.id_reg_w + exp_reg_loss * self.args.exp_reg_w + \
                            tex_reg_loss * self.args.tex_reg_w + photo_loss_val * self.args.rgb_loss_w
                    else:
                        rt_reg_loss = losses.get_l2(self.fvm.rot_tensor - rot_c) + losses.get_l2(self.fvm.trans_tensor - trans_c)
                        loss = lm_loss_val * self.args.lm_loss_w + exp_reg_loss * self.args.exp_reg_w + \
                            photo_loss_val * self.args.rgb_loss_w + rt_reg_loss * self.args.rt_reg_w
                    nonrigid_optimizer.zero_grad()
                    nonrigid_optimizer.backward(loss)
                    nonrigid_optimizer.step()
                    self.fvm.exp_tensor[self.fvm.exp_tensor < 0] *= 0
            
            # show data
            with jt.no_grad():
                if self.frame_ind == 0:
                    start_t = time.time()
                coeffs = self.fvm.get_packed_tensors().detach().clone()
                coeffs[:, self.fvm.id_dims + 8:self.fvm.id_dims + 10] = self.eyes_refine(coeffs[:, self.fvm.id_dims + 8:self.fvm.id_dims + 10])
                id_c, exp_c, tex_c, rot_c, gamma_c, trans_c, eye_c = self.fvm.split_coeffs(coeffs)
                if self.frame_ind == 0 and self.args.save_for_styleavatar:
                    np.savetxt(os.path.join(self.args.res_folder, 'id.txt'), id_c[0].numpy(), fmt='%.3f')
                    np.savetxt(os.path.join(self.args.res_folder, 'exp.txt'), exp_c[0].numpy(), fmt='%.3f')
                # for styleavatar test
                if self.args.id_folder is not None:
                    id_fisrt = jt.array(np.loadtxt(os.path.join(self.args.id_folder, 'id.txt')).astype(np.float32)[None, :], dtype=jt.float32)
                    exp_fisrt = jt.array(np.loadtxt(os.path.join(self.args.id_folder, 'exp.txt')).astype(np.float32)[None, :], dtype=jt.float32)
                    coeffs[:, :self.fvm.id_dims] += id_fisrt
                    # !!!only if the first frame is neutral expression!!!
                    if self.args.first_frame_is_neutral:
                        coeffs[:, self.fvm.id_dims:self.fvm.id_dims + self.fvm.exp_dims] += exp_fisrt
                if self.args.smooth:
                    if self.frame_ind == 0:
                        self.coeff_0 = coeffs.detach().clone()
                        self.coeff_1 = coeffs.detach().clone()
                        self.coeff_2 = coeffs.detach().clone()
                        out_queue.put(outimg)
                        self.align_last = align
                    else:
                        self.coeff_0 = self.coeff_1
                        self.coeff_1 = self.coeff_2
                        self.coeff_2 = coeffs.detach().clone()
                    coeffs = (self.coeff_0 + self.coeff_1 + self.coeff_2) / 3
                    align_tmp = align
                    align = self.align_last
                    self.align_last = align_tmp

                if self.args.save_for_styleavatar:
                    self.pred_dict = self.fvm(coeffs, render=True, surface=True, use_color=True, render_uv=True)
                else:
                    self.pred_dict = self.fvm(coeffs, render=True, surface=True, use_color=True)
                lms_proj = self.pred_dict['lms_proj'].numpy()
                lms_proj_center = jt.mean(lms_proj, dim=1)
                rendered_img_c = np.clip(self.pred_dict['rendered_img'].transpose((0, 2, 3, 1)).numpy(), 0, 255).astype(np.uint8)
                if self.args.save_for_styleavatar:
                    uv_img_c = np.clip(self.pred_dict['uv_img'].transpose((0, 2, 3, 1)).numpy(), 0, 255).astype(np.uint8)
                    drive_img = np.concatenate([align, rendered_img_c[0, :, :, :3], uv_img_c[0, :, :, :3]], axis=1)
                else:
                    drive_img = np.concatenate([align, rendered_img_c[0, :, :, :3]], axis=1)
                self.thread_lock.acquire()
                num_queue.put(frame_num)
                out_queue.put(outimg)
                if self.args.back is not None:
                    back_queue.put(backimg)
                image_queue.put(drive_img)
                self.queue_num += 1
                self.thread_lock.release()
            self.frame_ind += 1
            #self.offreader.crop_center += ((lms_proj[0, 168] / self.offreader.tar_size - 0.5) * self.offreader.half_length * 2).astype(np.int32)
            print(f'Speed:{(time.time() - start_t) / self.frame_ind:.4f}, ' + \
            f'{self.frame_ind:4} / {frame_num:4}, {3e3 * lm_loss_val.item():.4f}')
        self.thread_exit = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FaceVerse online tracker")

    parser.add_argument('--input', type=str, required=True,
                        help='input video path')
    parser.add_argument('--back', type=str, default=None,
                        help='background video path')
    parser.add_argument('--res_folder', type=str, required=True,
                        help='output directory')
    parser.add_argument('--id_folder', type=str, default=None,
                        help='id directory')
    parser.add_argument('--first_frame_is_neutral', action='store_true',
                        help='only if the first frame is neutral expression')
    parser.add_argument('--smooth', action='store_true',
                        help='smooth between 3 frames')
    parser.add_argument('--use_dr', action='store_true',
                        help='Can only be used on linux system.')
    parser.add_argument('--save_for_styleavatar', action='store_true',
                        help='Save images and parameters for styleavatar.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size is set to 1.')
    parser.add_argument('--skip_frames', type=int, default=0, 
                        help='Skip the first several frames.')
    parser.add_argument('--crop_size', type=int, default=1024,
                        help='size for output image.')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='size for output image.')
    parser.add_argument('--tar_size', type=int, default=256,
                        help='size for rendering window. We use a square window.')
    parser.add_argument('--lm_loss_w', type=float, default=1e3,
                        help='weight for landmark loss')
    parser.add_argument('--rgb_loss_w', type=float, default=1e-2,
                        help='weight for rgb loss')
    parser.add_argument('--id_reg_w', type=float, default=3e-2,
                        help='weight for id coefficient regularizer')
    parser.add_argument('--rt_reg_w', type=float, default=3e-2,
                        help='weight for rt regularizer')
    parser.add_argument('--exp_reg_w', type=float, default=3e-3,
                        help='weight for expression coefficient regularizer')
    parser.add_argument('--tex_reg_w', type=float, default=3e-3,
                        help='weight for texture coefficient regularizer')
    parser.add_argument('--tex_w', type=float, default=1,
                        help='weight for texture reflectance loss.')

    args = parser.parse_args()

    tracking = Tracking(args)
    tracking.start()

    os.makedirs(args.res_folder, exist_ok=True)
    if args.save_for_styleavatar:
        os.makedirs(os.path.join(args.res_folder, 'image'), exist_ok=True)
        os.makedirs(os.path.join(args.res_folder, 'uv'), exist_ok=True)
        os.makedirs(os.path.join(args.res_folder, 'render'), exist_ok=True)
        os.makedirs(os.path.join(args.res_folder, 'back'), exist_ok=True)
        import onnxruntime as ort
        sess = ort.InferenceSession('data/rvm_1024_1024_32.onnx')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    if args.save_for_styleavatar:
        tar_video = cv2.VideoWriter(os.path.join(args.res_folder, 'track.mp4'), fourcc, tracking.offreader.fps, (args.tar_size * 3, args.tar_size))
    else:
        tar_video = cv2.VideoWriter(os.path.join(args.res_folder, 'track.mp4'), fourcc, tracking.offreader.fps, (args.tar_size * 2, args.tar_size))
    #out_video = cv2.VideoWriter(os.path.join(args.res_folder, 'align.mp4'), fourcc, tracking.offreader.fps, (args.image_size, args.image_size))
    while True:
        if image_queue.empty():
            cv2.waitKey(15)
            continue
        tracking.thread_lock.acquire()
        fn = num_queue.get()
        out = out_queue.get()
        tar = image_queue.get()
        tracking.thread_lock.release()
        tracking.queue_num -= 1
        cv2.imshow('faceverse_offline', tar[:, :, ::-1])
        keyc = cv2.waitKey(1)
        if keyc == 27 or tracking.thread_exit == True:
            tracking.thread_exit = True
            break
        #out_video.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        tar_video.write(cv2.cvtColor(tar, cv2.COLOR_RGB2BGR))
        if args.save_for_styleavatar:
            cv2.imwrite(os.path.join(args.res_folder, 'image', str(fn).zfill(6) + '.png'), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.res_folder, 'render', str(fn).zfill(6) + '.png'), cv2.cvtColor(tar[:, args.tar_size:args.tar_size * 2], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.res_folder, 'uv', str(fn).zfill(6) + '.png'), cv2.cvtColor(tar[:, args.tar_size * 2:], cv2.COLOR_RGB2BGR))
            if args.back is None:
                if args.crop_size != 1024:
                    mask_in = cv2.resize(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), (1024, 1024))
                else:
                    mask_in = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                pha = sess.run(['out'], {'src': mask_in[None, :, :, :].astype(np.float32)})
                if args.crop_size != 1024:
                    mask_out = cv2.resize(pha[0][0, 0].astype(np.uint8), (args.crop_size, args.crop_size))
                else:
                    mask_out = pha[0][0, 0].astype(np.uint8)
            else:
                mask_out = back_queue.get()
            cv2.imwrite(os.path.join(args.res_folder, 'back', str(fn).zfill(6) + '.png'), mask_out)
        print('Write frames:', fn, 'still in queue:', tracking.queue_num)
    
    tracking.join()
    #out_video.release()
    tar_video.release()


