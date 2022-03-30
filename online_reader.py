import cv2
import numpy as np
import threading
import copy
from third_libs.OpenSeeFace.tracker import Tracker


class OnlineReader(threading.Thread):
    def __init__(self, camera_id, width, height):
        super(OnlineReader, self).__init__()
        self.camera_id = camera_id
        self.height, self.width = height, width#480, 640# 1080, 1920 480,640 600,800 720,1280 
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.lms = np.zeros((66, 2), dtype=np.uint8)
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



