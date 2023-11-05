## FaceVerse v3: Full head model & Realtime-tracking
Jittor-based tracking of faceverse model v3 (full head model, including two eyeballs) using [Jittor](https://github.com/Jittor/Jittor). And another windows exe version will be released in [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar), which is much faster.

### Requirements

- jittor==1.3.5.19 (Passed on 1.3.4 or 1.3.5, but failed on 1.3.8)
- mediapipe
- opencv-python
- numpy
- tqdm
- onnxruntime==1.12.1

**Note: the first compilation with Jittor before running the python scripts is quite slow and unstable. Please be patient and wait for responses. Use ctrl+c to quit the process if your process has been stacked and try again. (My experience: you need to try about 3 times for the first time)**

### FaceVerse v3 model

Full head model with two eyeballs, download: https://drive.google.com/file/d/1WrQ1UNMY30YAl8WxAbqVb6ZsPEQ_FHW4/view?usp=sharing

Baidu Netdisk: https://pan.baidu.com/s/1n81RmygpEU5tAqmVKZm3uw?pwd=8hls password: 8hls

Put the model in `faceversev3_jittor/data/faceverse_v3_6_s.npy`

For the preprocessing of [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar): [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) for save_for_styleavatar, download: https://drive.google.com/file/d/1544XkEO2rysSduJ55eBMX-nksW01NSNM/view?usp=sharing

Baidu Netdisk: https://pan.baidu.com/s/1ZZ7dEAE1EgoJVybuhBeY4A?pwd=5db6 password: 5db6

Put the model in `faceversev3_jittor/data/rvm_1024_1024_32.onnx`

**Note: rvm_1024_1024_32.onnx is generated from [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting), we thank the authors of RVM for their work. You can use this onnx file as shown in `rvm.py` (only for 1024x1024 input images). `--crop_size` should be 1024 for styleunet and 1536 for full styleavatar.**


### Usage

offline (including for the preprocessing of [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar)):

```
# for a single video 
# skip_frames is used to skip the first xx frames of the video, if there is no face in the first several frames
# --smooth to smooth the coeffs between 3 frames
python tracking_offline_cuda.py --input ../example/videos/test.mp4 --res_folder output/video --skip_frames 1 (--smooth)

# Preprocessing for StyleAvatar(https://github.com/LizhenWangT/StyleAvatar).
# crop_size should be 1024 for styleunet and 1536 for full styleavatar.
# --smooth to smooth the coeffs between 3 frames, add this if you need to test on the training video.
python tracking_offline_cuda.py --input ../example/videos/test.mp4 --res_folder output/video --save_for_styleavatar --crop_size 1024/1536 (--smooth)

# Testing for StyleAvatar(https://github.com/LizhenWangT/StyleAvatar)
# crop_size should be 1024 for styleunet and 1536 for full styleavatar.
# id_folder should be (path-to-the-training-folder) containing id.txt and exp.txt
# --first_frame_is_neutral can improve cross-person results only if the first frame is in a neutral expression. Otherwise, remove it.
# --smooth to smooth the coeffs between 3 frames
python tracking_offline_cuda.py --input ../example/videos/test.mp4 --res_folder output/test --save_for_styleavatar --smooth --crop_size 1024/1536 --id_folder output/video (--first_frame_is_neutral)

# for a image folder
python fit_imgs_offline_cuda.py --input ../example/images --res_folder output/images
```

online:

```
python tracking_online_cuda.py
```

**Now `--use_dr` can only be compiled succesfully on Linux system, which can improve the tracking quality. We also find that Jittor is much slower on Windows system.**
