# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import time
import pickle
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize
import os, subprocess
from keras.models import load_model
import xgboost as xgb
import signal
from contextlib import contextmanager
from matplotlib import animation as animation, pyplot as plt, cm


class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


try:
    with time_limit(200):
        parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
        parser.add_argument('--video', help='Path to video file.')
        parser.add_argument('--model')
        args = parser.parse_args()

# loading models
        estimator = load_pretrain_model('mobilenet_thin')
        if "XGB" in args.model:
            action_classifier = xgb.XGBClassifier()
            action_classifier.load_model(args.model)
        elif "pickle" in args.model:
            action_classifier = pickle.load(open(args.model,'rb'))
        else:
            action_classifier = load_model(args.model)

# FPS calculator
        realtime_fps = '0.0000'
        start_time = time.time()
        fps_interval = 1
        fps_count = 0
        run_timer = 0
        frame_count = 0

# webcam mode
        cap = choose_run_mode(args)
        video_writer = set_video_writer(cap, write_fps=int(7.0))
        fig, ax = plt.subplots()
        plt.ion()
        plt.xticks(rotation=45)


# # (for training)
# f = open('origin_data.txt', 'a+')
        while cv.waitKey(1) < 0:
            has_frame, show = cap.read()
            if has_frame:
                fps_count += 1
                frame_count += 1

        # pose estimation
                humans = estimator.inference(show)
        # get pose info
                pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
        # recognize the action framewise
                show, data = framewise_recognize(pose, action_classifier)
                Action = list(data.keys())
                Probability = list(data.values())
                bars = ax.bar(Action, Probability, color='green')
                for bar, height in zip(bars, Probability):
                    bar.set_height(height)
                plt.pause(5)
                plt.draw()
                height, width = show.shape[:2]
        # show fps
                if (time.time() - start_time) > fps_interval:
            # calculate interval
                    realtime_fps = fps_count / (time.time() - start_time)
                    fps_count = 0  # fps as 0
                    start_time = time.time()
                fps_label = 'FPS:{0:.2}'.format(realtime_fps)
                cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # show number of humans
                num_label = "Human: {0}".format(len(humans))
                cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # show how long it ran
                if frame_count == 1:
                    run_timer = time.time()
                run_time = time.time() - run_timer
                time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
                cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv.imshow('Action Recognition based on OpenPose', show)
                video_writer.write(show)
                # print("mobilenet_thin: " + str(frame_count))

        video_writer.release()
        cap.release()

except TimeoutException as e:
    print("Timed out!")

# f.close()
