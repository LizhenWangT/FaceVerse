import cv2
from PIL import Image
import numpy as np
import threading
import copy
import time
import os
from util_functions import get_length
import mediapipe as mp


class OnlineReader(threading.Thread):
    def __init__(self, camera_id, width, height, tar_size):
        super(OnlineReader, self).__init__()
        self.camera_id = camera_id
        self.height, self.width = height, width#480, 640# 1080, 1920 480,640 600,800 720,1280 
        self.tar_size = tar_size
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.frame_num = 0
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(3, width)
        self.cap.set(4, height)
        fourcc= cv2.VideoWriter_fourcc('M','J','P','G')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.thread_exit = False
        self.thread_lock = threading.Lock()
        self.length_scale = 1.0
        self.face_tracker = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_data(self):
        return copy.deepcopy(self.frame), copy.deepcopy(self.frame_num)

    def run(self):
        while not self.thread_exit:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                if self.frame_num == 0:
                    results = self.face_tracker.process(frame)
                    if not results.multi_face_landmarks:
                        print('No face detected in online reader!')
                        self.detected = False
                        time.sleep(0.015)
                        continue
                    lms = results.multi_face_landmarks[0]
                    lms = np.array([(lmk.x, lmk.y) for lmk in lms.landmark])
                    lms[:, 0] *= frame.shape[1]
                    lms[:, 1] *= frame.shape[0]
                    lms = lms .astype(np.int32)
                    self.border = 500
                    self.half_length = int(get_length(lms) * self.length_scale)
                    self.crop_center = lms[197].copy() + self.border
                    print('First frame:', self.half_length, self.crop_center)
        
                frame_b = cv2.copyMakeBorder(frame, self.border, self.border, self.border, self.border, cv2.BORDER_CONSTANT, value=0)
                frame_b = Image.fromarray(frame_b[self.crop_center[1] - self.half_length:self.crop_center[1] + self.half_length, 
                                            self.crop_center[0] - self.half_length:self.crop_center[0] + self.half_length])
                align = np.asarray(frame_b.resize((self.tar_size, self.tar_size), Image.ANTIALIAS))
                self.thread_lock.acquire()
                self.frame_num += 1
                self.frame = align
                self.thread_lock.release()
            else:
                self.thread_exit = True
        self.cap.release()


class OnlineDetecder(threading.Thread):
    def __init__(self, camera_id, width, height, tar_size, batch_size):
        super(OnlineDetecder, self).__init__()
        self.onreader = OnlineReader(camera_id, width, height, tar_size)
        self.detected = False
        self.thread_exit = False
        self.tar_size = tar_size
        self.thread_lock = threading.Lock()
        self.batch_size = batch_size
        self.frame = np.zeros((batch_size, tar_size, tar_size, 3), dtype=np.uint8)
        #self.lms = np.zeros((batch_size, 66, 2), dtype=np.int64)
        self.lms = np.zeros((batch_size, 478, 2), dtype=np.int64)
        self.frame_num = 0
        self.face_tracker = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_data(self):
        return copy.deepcopy(self.frame), copy.deepcopy(self.lms), copy.deepcopy(self.frame_num)

    def run(self):
        self.onreader.start()
        while not self.thread_exit:
            self.onreader.thread_lock.acquire()
            align, frame_num = self.onreader.get_data()
            self.onreader.thread_lock.release()
            if frame_num <= self.frame_num:
                #print('wait frame')
                time.sleep(0.015)
                continue
            results = self.face_tracker.process(align)
            if not results.multi_face_landmarks:
                print('No face detected in online reader!')
                self.detected = False
                time.sleep(0.015)
                continue
            lms = results.multi_face_landmarks[0]
            lms = np.array([(lmk.x, lmk.y) for lmk in lms.landmark])
            lms[:, 0] *= align.shape[1]
            lms[:, 1] *= align.shape[0]
            
            self.detected = True
            
            self.thread_lock.acquire()
            self.frame[self.frame_num % self.batch_size] = align
            self.lms[self.frame_num % self.batch_size] = lms
            self.frame_num += 1#frame_num
            self.thread_lock.release()
            #print(self.frame_num, frame_num)
        self.onreader.thread_exit = True


class OfflineReader:
    def __init__(self, path, tar_size, image_size, skip_frames=0):
        self.skip_frames = skip_frames
        self.tar_size = tar_size
        self.image_size = image_size
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num = 0
        self.length_scale = 1.0
        self.border = 500
        self.detected = False
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.face_tracker0 = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_tracker = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_data(self):
        ret, frame = self.cap.read()
        if ret:
            while self.frame_num < self.skip_frames:
                _, frame = self.cap.read()
                self.frame_num += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not self.detected:
                results = self.face_tracker0.process(frame)
                if not results.multi_face_landmarks:
                    print('No face detected in online reader!')
                    return False, False, [], [], []
                lms = results.multi_face_landmarks[0]
                lms = np.array([(lmk.x, lmk.y) for lmk in lms.landmark])
                lms[:, 0] *= frame.shape[1]
                lms[:, 1] *= frame.shape[0]
                lms = lms.astype(np.int32)
                self.half_length = int(get_length(lms) * self.length_scale)
                self.crop_center = lms[197] + self.border
                print('First frame:', self.half_length, self.crop_center)
            
            frame_b = cv2.copyMakeBorder(frame, self.border, self.border, self.border, self.border, cv2.BORDER_CONSTANT, value=0)
            frame_b = Image.fromarray(frame_b[self.crop_center[1] - self.half_length:self.crop_center[1] + self.half_length, 
                                        self.crop_center[0] - self.half_length:self.crop_center[0] + self.half_length])
            align = np.asarray(frame_b.resize((self.tar_size, self.tar_size), Image.ANTIALIAS))
            outimg = np.asarray(frame_b.resize((self.image_size, self.image_size), Image.ANTIALIAS))
            
            results = self.face_tracker.process(align)
            if not results.multi_face_landmarks:
                print('No face detected in offline reader!')
                return False, False, [], [], []
            
            lms = np.zeros((478, 2), dtype=np.int64)
            for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                #print(idx, landmark.x)
                #if idx < 468:
                lms[idx, 0] = int(landmark.x * self.tar_size)
                lms[idx, 1] = int(landmark.y * self.tar_size)
            self.frame_num += 1
            self.detected = True
            return True, align, lms, outimg, self.frame_num
        else:
            self.cap.release()
            print('Reach the end of the video')
            return False, True, [], [], []


class ImageReader:
    def __init__(self, path, image_size):
        self.path = path
        self.imagelist = os.listdir(path)
        self.num_frames = len(self.imagelist)
        self.frame_num = 0
        self.image_size = image_size
        
        self.face_tracker = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_data(self):
        if self.frame_num == self.num_frames:
            print('Reach the end of the folder')
            return False, True, [], [], []

        frame = cv2.imread(os.path.join(self.path, self.imagelist[self.frame_num]), -1)[:, :, :3]
        frame = frame[:, :, ::-1]
        height, width = frame.shape[:2]
        if height != self.image_size or width != self.image_size:
            frame = cv2.resize(frame, (self.image_size, self.image_size))
        #frame = np.concatenate([frame, frame[-176:]], axis=0)
        
        # 3 times for beter detection
        results = self.face_tracker.process(frame)
        results = self.face_tracker.process(frame)
        results = self.face_tracker.process(frame)
        if not results.multi_face_landmarks:
            print('No face detected in ' + self.imagelist[self.frame_num])
            self.frame_num += 1
            return False, False, [], [], []
        lms = np.zeros((478, 2), dtype=np.int64)
        for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
            #print(idx, landmark.x)
            #if idx < 468:
            lms[idx, 0] = int(landmark.x * self.image_size)
            lms[idx, 1] = int(landmark.y * self.image_size)
        self.frame_num += 1
        return True, frame, lms, self.frame_num, self.imagelist[self.frame_num - 1]

