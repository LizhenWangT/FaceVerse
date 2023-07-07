from faceverse_cuda.FaceVerseModel import FaceVerseModel
import numpy as np

def get_faceverse(**kargs):
    model_path = 'data/faceverse_v3_6_s.npy'
    faceverse_dict = np.load(model_path, allow_pickle=True).item()
    faceverse_model = FaceVerseModel(faceverse_dict, **kargs)
    return faceverse_model, faceverse_dict

