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

from data_reader import OnlineDetecder
from util_functions import get_length, ply_from_array_color


image_queue = Queue()


class Tracking(threading.Thread):
    def __init__(self, args):
        super(Tracking, self).__init__()
        self.args = args
        self.fvm, self.fvd = get_faceverse(batch_size=self.args.batch_size, focal=int(1315), img_size=self.args.tar_size)
        self.lm_weights = losses.get_lm_weights()
        self.onreader = OnlineDetecder(camera_id=0, width=800, height=600, tar_size=self.args.tar_size, batch_size=self.args.batch_size)
        self.onreader.start()
        self.thread_lock = threading.Lock()
        self.frame_ind = 0
        self.thread_exit = False
        self.queue_num = 0
        self.frame_last = 0

    def run(self):
        while not self.thread_exit:
            # load data
            self.onreader.thread_lock.acquire()
            align, lms_detect, frame_num = self.onreader.get_data()
            self.onreader.thread_lock.release()
            if frame_num <= self.frame_ind * self.args.batch_size or frame_num % self.args.batch_size != 0 or frame_num == self.frame_last:
                time.sleep(0.01)
                continue
            self.frame_last = frame_num

            lms = jt.array(lms_detect, dtype=jt.float32).stop_grad()
            img_tensor = jt.array(align, dtype=jt.float32).stop_grad().transpose((0, 3, 1, 2))

            if self.frame_ind == 0:
                num_iters_rf = 1000
                num_iters_nrf = 200
                rigid_optimizer = jt.optim.Adam([self.fvm.rot_tensor, self.fvm.trans_tensor, self.fvm.id_tensor, self.fvm.exp_tensor, self.fvm.eye_tensor], 
                                                    lr=1e-2, betas=(0.8, 0.95))
                nonrigid_optimizer = jt.optim.Adam([self.fvm.id_tensor, self.fvm.gamma_tensor, self.fvm.tex_tensor,
                                                    self.fvm.rot_tensor, self.fvm.trans_tensor, self.fvm.eye_tensor], lr=5e-3, betas=(0.5, 0.9))
            else:
                #lms_center = jt.mean(lms, dim=1)
                #self.fvm.trans_tensor[:, :2] -= (lms_center - lms_proj_center) * self.fvm.trans_tensor[:, 2:3] / self.fvm.focal * 0.5
                rigid_optimizer = jt.optim.Adam([self.fvm.rot_tensor, self.fvm.trans_tensor, self.fvm.exp_tensor, self.fvm.eye_tensor], 
                                                    lr=1e-2, betas=(0.5, 0.9))
                nonrigid_optimizer = jt.optim.Adam([self.fvm.exp_tensor, self.fvm.gamma_tensor, 
                                                    self.fvm.rot_tensor, self.fvm.trans_tensor, self.fvm.eye_tensor], lr=5e-3, betas=(0.5, 0.9))
                num_iters_rf = 25
                num_iters_nrf = 10
            
            # fitting using only landmarks
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

                if self.frame_ind == 0:
                    id_reg_loss = losses.get_l2(self.fvm.id_tensor)
                    tex_reg_loss = losses.get_l2(self.fvm.tex_tensor)
                    loss = lm_loss_val * self.args.lm_loss_w * 0.3 + id_reg_loss * self.args.id_reg_w + exp_reg_loss * self.args.exp_reg_w + \
                        tex_reg_loss * self.args.tex_reg_w + photo_loss_val * self.args.rgb_loss_w
                else:
                    rt_reg_loss = losses.get_l2(self.fvm.rot_tensor - rot_c) + losses.get_l2(self.fvm.trans_tensor - trans_c)
                    loss = lm_loss_val * self.args.lm_loss_w * 0.3 + exp_reg_loss * self.args.exp_reg_w + \
                           photo_loss_val * self.args.rgb_loss_w + rt_reg_loss * self.args.rt_reg_w
                
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
                lms_proj = self.pred_dict['lms_proj'].numpy()
                #lms_proj_center = jt.mean(lms_proj, dim=1)
                rendered_img_c = np.clip(self.pred_dict['rendered_img'].transpose((0, 2, 3, 1)).numpy(), 0, 255)[:, :, :, :3].astype(np.uint8)
                #mask_r = (rendered_img_c[:, :, :, 3:4] > 0).astype(np.uint8)
                #render = align * (1 - mask_r) + rendered_img_c[:, :, :, :3] * mask_r
                #for imgi in range(self.args.batch_size):
                #    for i in range(468, 475):
                #        cv2.circle(rendered_img_c[imgi], (int(lms_proj[imgi, i, 0]), int(lms_proj[imgi, i, 1])), 1, (0, 0, 255), -1)
                #for imgi in range(self.args.batch_size):
                #    for i in range(468, 475):
                #        cv2.circle(align[imgi], (lms_detect[imgi, i, 0], lms_detect[imgi, i, 1]), 1, (0, 0, 255), -1)
                drive_img = np.concatenate([align, rendered_img_c], axis=2)
                self.thread_lock.acquire()
                for imgi in range(self.args.batch_size):
                    image_queue.put(drive_img[imgi])
                    self.queue_num += 1
                self.thread_lock.release()
            
            self.frame_ind += 1
            print(f'Speed:{(time.time() - start_t) / self.frame_ind / self.args.batch_size:.4f}, ' + \
            f'{self.frame_ind * self.args.batch_size:4} / {frame_num:4}')
        self.onreader.thread_exit = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FaceVerse online tracker")

    parser.add_argument('--batch_size', type=int, default=3,
                        help='batch_size.')
    parser.add_argument('--save', action='store_true',
                        help='save video.')
    parser.add_argument('--tar_size', type=int, default=256,
                        help='size for rendering window. We use a square window.')
    parser.add_argument('--lm_loss_w', type=float, default=1e3,
                        help='weight for landmark loss')
    parser.add_argument('--rgb_loss_w', type=float, default=1e-2,
                        help='weight for rgb loss')
    parser.add_argument('--id_reg_w', type=float, default=3e-3,
                        help='weight for id coefficient regularizer')
    parser.add_argument('--rt_reg_w', type=float, default=3e-2,
                        help='weight for rt regularizer')
    parser.add_argument('--exp_reg_w', type=float, default=1e-3,
                        help='weight for expression coefficient regularizer')
    parser.add_argument('--tex_reg_w', type=float, default=3e-5,
                        help='weight for texture coefficient regularizer')
    parser.add_argument('--tex_w', type=float, default=1,
                        help='weight for texture reflectance loss.')

    args = parser.parse_args()

    tracking = Tracking(args)
    tracking.start()
    
    scale = 1
    imageshow = np.zeros((args.tar_size * scale, args.tar_size * 2 * scale, 3), np.uint8)
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = 25
        tar_video = cv2.VideoWriter('track.avi', fourcc, fps, (args.tar_size * 2 * scale, args.tar_size * 2 * scale))
    while True:
        if image_queue.empty():
            cv2.imshow('FaceVerse Tracking', imageshow[:, :, ::-1])
            keyc = cv2.waitKey(5)
            if keyc == 27:
                tracking.thread_exit = True
                break
            continue
        imageshow = image_queue.get()
        tracking.queue_num -= 1
        imageshow = cv2.resize(imageshow, (args.tar_size * 2 * scale, args.tar_size * scale))
        cv2.imshow('FaceVerse Tracking', imageshow[:, :, ::-1])
        if args.save:
            tar_video.write(imageshow[:, :, ::-1])
        wait_time = max(30 - tracking.queue_num * 5, 1)
        keyc = cv2.waitKey(wait_time) & 0xFF
        if keyc == ord('q'):
            tracking.thread_exit = True
            break
        elif keyc == ord('z'):
            tracking.onreader.onreader.half_length = int(tracking.onreader.onreader.half_length * 1.01)
        elif keyc == ord('x'):
            tracking.onreader.onreader.half_length = int(tracking.onreader.onreader.half_length / 1.01)
        elif keyc == ord('a'):
            tracking.onreader.onreader.crop_center[0] += 1
        elif keyc == ord('d'):
            tracking.onreader.onreader.crop_center[0] -= 1
        elif keyc == ord('w'):
            tracking.onreader.onreader.crop_center[1] += 1
        elif keyc == ord('s'):
            tracking.onreader.onreader.crop_center[1] -= 1
        elif keyc == ord('f'):
            vertices = tracking.pred_dict['vertices'][0].detach().numpy()
            colors = tracking.pred_dict['colors'][0].detach().numpy()
            colors = np.clip(colors, 0, 255).astype(np.uint8)
            ply_from_array_color(vertices, colors, tracking.fvd['tri'], 'test.ply')
    
    tracking.join()
    if args.save:
        tar_video.release()
    cv2.destroyAllWindows()


