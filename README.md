# OpenPose-Action-Recognition-Model

A skeleton-based real-time online action recognition project, classifying and recognizing base on framewise joints, which can be used for safety monitoring..   


------
## Introduction
*The **pipline** of this work is:*   
 - Realtime pose estimation by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose);   
 - Online human tracking for multi-people scenario by [DeepSort algorithm](https://github.com/nwojke/deep_sortv);   
 - Action recognition with DNN for each person based on single framewise joints detected from Openpose.


------
## Dependencies
 - python >= 3.5
 - Opencv >= 3.4.1   
 - sklearn
 - tensorflow & keras
 - numpy & scipy 
 - pathlib
 
 
------
## Usage
 - Put your test videos into a directory and copy the directory path.
 - Open your terminal (command prompt), and change the directory to where the action recognition directory is located. Do this by copying the path of where the action recognition  directory is located, then run the following command in the terminal:
$ cd <path>
 -  In the same command line window from the previous section, run the following line:
$ python3 main.py --video <video_name> --model <model_name>
 - After running the steps stated above, the results will soon show up on your screen.

## Explanation of the Parameters
● --video
File path to the input video. The path should exist and the file extension should be included as well. For example, “punching.mp4”, instead of “punching”
● --model
File path to the saved trained model for action recognition. There is no default value for this parameter, therefore a model must be used. The file extension should be included.

------
## Training with own dataset
 - prepare data(actions) by running `openpose_coco.py`.
 - transforming the `.txt` to `.csv`, you can use EXCEL to do this.
 - do the training with the `traing.py` in `Action/training/`, remember to ***change the action_enum and output-layer of model***.
 

-------
## Note
 - Action recognition in this work is framewise based, so it's technically "**Pose recognition**" to be exactly;   
 - Action is actually a dynamic motion which consists of sequential static poses, therefore classifying framewisely is not a good solution.
 - Considering of using ***RNN(LSTM) model*** to classify actions with dynamic sequential joints data is the next step to improve this project.


------
## Acknowledge
Thanks to the following awesome works:    
 - [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)   
 - [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3)    
 - [Real-Time-Action-Recognition](https://github.com/TianzhongSong/Real-Time-Action-Recognition)
