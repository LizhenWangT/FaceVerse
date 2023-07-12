import onnxruntime as ort
import os
import cv2
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RVM")
    parser.add_argument("--path", type=str, default='D:/test_case', help="path to the image dataset")
    parser.add_argument("--ckpt", type=str, default='data/rvm_1024_1024_32.onnx', help="path to the checkpoints to resume training")
    args = parser.parse_args()

    sess = ort.InferenceSession(args.ckpt)
    os.makedirs(os.path.join(args.path, 'back'), exist_ok=True)
    for name in os.listdir(os.path.join(args.path, 'image')):
        img = cv2.imread(os.path.join(args.path, 'image', name))[None, :, :, :].astype(np.float32)
        pha = sess.run(['out'], {
            'src': img
        })
        print(name)
        cv2.imshow('rvm', pha[0][0, 0].astype(np.uint8))
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(args.path, 'back', name), pha[0][0, 0].astype(np.uint8))


