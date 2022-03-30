import cv2
import numpy as np
import copy
from third_libs.OpenSeeFace.tracker import Tracker


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


