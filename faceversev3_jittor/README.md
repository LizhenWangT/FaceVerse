## FaceVerse v3: Full head model & Realtime-tracking
Jittor-based tracking of faceverse model v3 (full head model, including two eyeballs) using [Jittor](https://github.com/Jittor/Jittor). And another windows exe version will be released in [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar), which is much faster.

### Requirements

- jittor
- mediapipe
- opencv-python
- numpy
- tqdm

**Note: the first compilation with Jittor before running the python scripts is quite slow and unstable. Please be patient and wait for responses. Use ctrl+c to quit the process if your process has been stacked and try again. (My experience: you need to try about 3 times for the first time)**

### FaceVerse v3 model

Full head model with two eyeballs, download: https://drive.google.com/file/d/1WrQ1UNMY30YAl8WxAbqVb6ZsPEQ_FHW4/view?usp=sharing

Put the model in `faceversev3_jittor/data/faceverse_v3_6_s.npy`


### Usage

offline:

```
# for a single video 
# skip_frames is used to skip the first xx frames of the video, if there is no face in the first several frames
python tracking_offline_cuda.py --input ../example/videos/test.mp4 --res_folder output/video --skip_frames 1
# for a single video and save data for StyleAvatar (https://github.com/LizhenWangT/StyleAvatar)
python tracking_offline_cuda.py --input ../example/videos/test.mp4 --res_folder output/video --save_for_styleavatar
# for a image folder
python fit_imgs_offline_cuda.py --input ../example/images --res_folder output/images
```

online:

```
python tracking_online_cuda.py
```

**Now `--use_dr` can only be compiled succesfully on Linux system, which can improve the tracking quality. We also find that Jittor is much slower on Windows system.**
