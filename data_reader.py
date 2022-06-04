import cv2
import numpy as np
import threading
import copy
import os
from third_libs.OpenSeeFace.tracker import Tracker


class OnlineReader(threading.Thread):
    def __init__(self, camera_id, width, height):
        super(OnlineReader, self).__init__()
        self.camera_id = camera_id
        self.height, self.width = height, width#480, 640# 1080, 1920 480,640 600,800 720,1280 
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.lms = np.zeros((66, 2), dtype=np.int64)
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(3, width)
        self.cap.set(4, height)
        fourcc= cv2.VideoWriter_fourcc('M','J','P','G')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.thread_lock = threading.Lock()
        self.thread_exit = False
        self.frame_num = 0
        self.tracker = Tracker(width, height, threshold=None, max_threads=1,
                              max_faces=1, discard_after=10, scan_every=30, 
                              silent=True, model_type=4, model_dir='third_libs/OpenSeeFace/models', no_gaze=True, detection_threshold=0.6, 
                              use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=0)

    def get_data(self):
        return copy.deepcopy(self.frame), copy.deepcopy(self.lms), copy.deepcopy(self.frame_num)

    def run(self):
        while not self.thread_exit:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                preds = self.tracker.predict(frame)
                if len(preds) == 0:
                    print('No face detected in online reader!')
                    continue
                # try more times in the fisrt frame for better landmarks
                if self.frame_num == 0:
                    for _ in range(3):
                        preds = self.tracker.predict(frame)
                        if len(preds) == 0:
                            print('No face detected in offline reader!')
                            continue
                lms = (preds[0].lms[:66, :2].copy() + 0.5).astype(np.int64)
                lms = lms[:, [1, 0]]
                self.thread_lock.acquire()
                self.frame_num += 1
                self.frame = frame
                self.lms = lms
                self.thread_lock.release()
            else:
                self.thread_exit = True
        self.cap.release()


class OfflineReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num = 0
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.tracker = Tracker(self.width, self.height, threshold=None, max_threads=1,
                              max_faces=1, discard_after=10, scan_every=30, 
                              silent=True, model_type=4, model_dir='third_libs/OpenSeeFace/models', no_gaze=True, detection_threshold=0.6, 
                              use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=0)

    def get_data(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            preds = self.tracker.predict(frame)
            if len(preds) == 0:
                print('No face detected in offline reader!')
                return False, False, [], []
            # try more times in the fisrt frame for better landmarks
            if self.frame_num == 0:
                for _ in range(3):
                    preds = self.tracker.predict(frame)
                    if len(preds) == 0:
                        print('No face detected in offline reader!')
                        return False, False, [], []
            lms = (preds[0].lms[:66, :2].copy() + 0.5).astype(np.int64)
            lms = lms[:, [1, 0]]
            self.frame_num += 1
            return True, frame, lms, self.frame_num
        else:
            self.cap.release()
            print('Reach the end of the video')
            return False, True, [], []


class ImageReader:
    def __init__(self, path):
        self.path = path
        self.imagelist = os.listdir(path)
        self.num_frames = len(self.imagelist)
        self.frame_num = 0

    def get_data(self):
        if self.frame_num == self.num_frames:
            print('Reach the end of the folder')
            return False, True, [], []

        frame = cv2.imread(os.path.join(self.path, self.imagelist[self.frame_num]), -1)[:, :, :3]
        frame = frame[:, :, ::-1]
        height, width = frame.shape[:2]
        tracker = Tracker(width, height, threshold=None, max_threads=1,
                        max_faces=1, discard_after=10, scan_every=30, 
                        silent=True, model_type=4, model_dir='third_libs/OpenSeeFace/models', no_gaze=True, detection_threshold=0.6, 
                        use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=0)
        preds = tracker.predict(frame)
        if len(preds) == 0:
            print('No face detected in ' + self.imagelist[self.frame_num])
            self.frame_num += 1
            return False, False, [], []
        # try more times in the fisrt frame for better landmarks
        for _ in range(3):
            preds = tracker.predict(frame)
            if len(preds) == 0:
                print('No face detected in ' + self.imagelist[self.frame_num])
                self.frame_num += 1
                return False, False, [], []
        lms = (preds[0].lms[:66, :2].copy() + 0.5).astype(np.int64)
        lms = lms[:, [1, 0]]
        self.frame_num += 1
        return True, frame, lms, self.frame_num




