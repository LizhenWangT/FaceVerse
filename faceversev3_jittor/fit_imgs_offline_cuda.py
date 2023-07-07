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

from data_reader import ImageReader
from util_functions import get_length, ply_from_array


num_queue = Queue()
image_queue = Queue()
name_queue = Queue()
vs_queue = Queue()
param_queue = Queue()


class Tracking(threading.Thread):
    def __init__(self, args):
        super(Tracking, self).__init__()
        self.args = args
        self.fvm, self.fvd = get_faceverse(batch_size=self.args.batch_size, focal=int(1315 / 512 * self.args.tar_size), img_size=self.args.tar_size)
        self.lm_weights = losses.get_lm_weights()
        self.offreader = ImageReader(args.input, args.image_size)
        self.thread_lock = threading.Lock()
        self.frame_ind = 0
        self.thread_exit = False
        self.queue_num = 0

    def run(self):
        while not self.thread_exit:
            # load data
            detected, align, lms_detect, frame_num, frame_name = self.offreader.get_data()
            if not detected:
                if not align:
                    continue
                else:
                    self.thread_exit = True
                    break
            lms = jt.array(lms_detect[None, :, :], dtype=jt.float32).stop_grad()
            img_tensor = jt.array(align[None, :, :, :], dtype=jt.float32).stop_grad().transpose((0, 3, 1, 2))

            num_iters_rf = 500
            num_iters_nrf = 200
            rigid_optimizer = jt.optim.Adam([self.fvm.rot_tensor, self.fvm.trans_tensor, self.fvm.id_tensor, self.fvm.exp_tensor, self.fvm.eye_tensor], 
                                                lr=1e-2, betas=(0.8, 0.95))
            nonrigid_optimizer = jt.optim.Adam([self.fvm.id_tensor, self.fvm.gamma_tensor, self.fvm.tex_tensor, self.fvm.exp_tensor,
                                                self.fvm.rot_tensor, self.fvm.trans_tensor, self.fvm.eye_tensor], lr=5e-3, betas=(0.5, 0.9))
            
            # fitting using only landmarks
            for i in range(num_iters_rf):
                pred_dict = self.fvm(self.fvm.get_packed_tensors(), render=False)
                lm_loss_val = losses.lm_loss(pred_dict['lms_proj'], lms, self.lm_weights, img_size=self.args.tar_size)
                exp_reg_loss = losses.get_l2(self.fvm.exp_tensor)
                id_reg_loss = losses.get_l2(self.fvm.id_tensor)
                loss = lm_loss_val * self.args.lm_loss_w + id_reg_loss * self.args.id_reg_w + exp_reg_loss * self.args.exp_reg_w
                
                rigid_optimizer.zero_grad()
                rigid_optimizer.backward(loss)
                rigid_optimizer.step()
            
                self.fvm.exp_tensor[self.fvm.exp_tensor < 0] *= 0
            '''
            # fitting with differentiable rendering
            for i in range(num_iters_nrf):
                pred_dict = self.fvm(self.fvm.get_packed_tensors(), render=True)
                rendered_img = pred_dict['rendered_img']
                lms_proj = pred_dict['lms_proj']
                face_texture = pred_dict['colors']

                lm_loss_val = losses.lm_loss(lms_proj, lms, self.lm_weights,img_size=self.args.tar_size)
                photo_loss_val = losses.photo_loss(rendered_img[:, :3], img_tensor)
                exp_reg_loss = losses.get_l2(self.fvm.exp_tensor)

                id_reg_loss = losses.get_l2(self.fvm.id_tensor)
                tex_reg_loss = losses.get_l2(self.fvm.tex_tensor)
                loss = lm_loss_val * self.args.lm_loss_w * 0.3 + id_reg_loss * self.args.id_reg_w + exp_reg_loss * self.args.exp_reg_w + \
                    tex_reg_loss * self.args.tex_reg_w + photo_loss_val * self.args.rgb_loss_w
                
                nonrigid_optimizer.zero_grad()
                nonrigid_optimizer.backward(loss)
                nonrigid_optimizer.step()
            
                self.fvm.exp_tensor[self.fvm.exp_tensor < 0] *= 0
            '''
            
            # show data
            with jt.no_grad():
                if self.frame_ind == 0:
                    start_t = time.time()
                coeffs = self.fvm.get_packed_tensors().detach().clone()
                id_c, exp_c, tex_c, rot_c, gamma_c, trans_c, eye_c = self.fvm.split_coeffs(coeffs)
                #pred_dict = self.fvm(self.fvm.get_packed_tensors(), render=True, surface=True, use_color=False)
                #rendered_img_r = np.clip(pred_dict['rendered_img'].transpose((0, 2, 3, 1)).numpy(), 0, 255).astype(np.uint8)
                self.pred_dict = self.fvm(coeffs, render=True, surface=True, use_color=True)
                lms_proj = self.pred_dict['lms_proj']
                lms_proj_center = jt.mean(lms_proj, dim=1)
                rendered_img_c = np.clip(self.pred_dict['rendered_img'].transpose((0, 2, 3, 1)).numpy(), 0, 255).astype(np.uint8)
                #mask_r = (rendered_img_c[:, :, :, 3:4] > 0).astype(np.uint8)
                #render = align * (1 - mask_r) + rendered_img_c[:, :, :, :3] * mask_r
                drive_img = np.concatenate([align, rendered_img_c[0, :, :, :3]], axis=1)
                self.thread_lock.acquire()
                num_queue.put(frame_num)
                image_queue.put(drive_img)
                name_queue.put(frame_name)
                param_queue.put(coeffs[0].numpy())
                vs_queue.put(self.pred_dict['vertices'][0].detach().numpy())
                self.queue_num += 1
                self.thread_lock.release()
            self.frame_ind += 1
            print(f'Speed:{(time.time() - start_t) / self.frame_ind:.4f}, ' + \
            f'{self.frame_ind:4} / {frame_num:4}, {3e3 * lm_loss_val.item():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FaceVerse online tracker")

    parser.add_argument('--input', type=str, required=True,
                        help='input video path')
    parser.add_argument('--res_folder', type=str, required=True,
                        help='output directory')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='batch_size.')
    parser.add_argument('--save', action='store_true',
                        help='save video.')
    parser.add_argument('--lm_loss_w', type=float, default=1e3,
                        help='weight for landmark loss')
    parser.add_argument('--rgb_loss_w', type=float, default=1e-2,
                        help='weight for rgb loss')
    parser.add_argument('--id_reg_w', type=float, default=3e-3,
                        help='weight for id coefficient regularizer')
    parser.add_argument('--rt_reg_w', type=float, default=3e-1,
                        help='weight for rt regularizer')
    parser.add_argument('--exp_reg_w', type=float, default=8e-3,
                        help='weight for expression coefficient regularizer')
    parser.add_argument('--tex_reg_w', type=float, default=3e-5,
                        help='weight for texture coefficient regularizer')
    parser.add_argument('--tex_w', type=float, default=1,
                        help='weight for texture reflectance loss.')
    parser.add_argument('--skip_frames', type=int, default=0, 
                        help='Skip the first several frames.')
    parser.add_argument('--image_size', type=int, default=512,
                        help='size for output image.')
    parser.add_argument('--tar_size', type=int, default=512,
                        help='size for rendering window. We use a square window.')

    args = parser.parse_args()

    tracking = Tracking(args)
    tracking.start()
    
    tri = tracking.fvd['tri'].copy()
    os.makedirs(args.res_folder, exist_ok=True)
    os.makedirs(os.path.join(args.res_folder, 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.res_folder, 'param'), exist_ok=True)
    
    while True:
        if image_queue.empty():
            if tracking.thread_exit:
                break
            cv2.waitKey(15)
            continue
        tracking.thread_lock.acquire()
        fn = num_queue.get()
        tar = image_queue.get()
        filename = name_queue.get()
        vs = vs_queue.get()
        param = param_queue.get()
        tracking.thread_lock.release()
        tracking.queue_num -= 1
        cv2.imwrite(os.path.join(args.res_folder, 'render', filename), tar[:, :, ::-1])
        np.save(os.path.join(args.res_folder, 'param', filename.split('.')[0] + '.npy'), param)
        print('Write frames:', fn, 'still in queue:', tracking.queue_num)
        
        if False:
            #colors = tracking.pred_dict['colors'].detach().squeeze().numpy()
            #colors = np.clip(colors, 0, 255).astype(np.uint8)
            #ply_from_array_color(vertices, colors, fvd['tri'], os.path.join(args.res_folder, filename.split('.')[0] + '.ply'))
            ply_from_array(vs, tracking.fvd['tri'], os.path.join(args.res_folder, filename.split('.')[0] + '.ply'))
    
    tracking.join()


