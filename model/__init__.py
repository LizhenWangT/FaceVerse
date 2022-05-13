from model.FaceVerseModel import FaceVerseModel
import numpy as np

def get_faceverse(version, **kargs):
    if version == 0:
        model_path = 'data/faceverse_base_v0.npy'
    elif version == 1:
        model_path = 'data/faceverse_base_v1.npy'
    elif version == 2:
        model_path = 'data/faceverse_simple_v2.npy'
    faceverse_dict = np.load(model_path, allow_pickle=True).item()
    faceverse_model = FaceVerseModel(faceverse_dict, **kargs)
    return faceverse_model, faceverse_dict

