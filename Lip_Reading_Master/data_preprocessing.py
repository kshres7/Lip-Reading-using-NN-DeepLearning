#!/usr/bin/env python
# coding: utf-8

# DATA PRE-PROCESSING - LRW Dataset

# In[4]:


import glob
import numpy as np
import cv2
import imutils
import os


# In[5]:


# dataset soure destination 
video_path = 'LRW/lipread_mp4'

# folder to save the preprocessed samples
NPY_FOLDER = os.path.join("NPY", "npy_28")

# set the sizes of mouth region (ROI) -> input shape
WIDTH = 24
HEIGHT = 32
DEPTH = 28

# print to debug
debug = False

# Haar cascade classifiers - frontal face, profile face and mouth detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')


# In[6]:


# select words for LRW dataset to train
train_words = ["ABOUT", "ACCESS", "ALLOW", "BANKS", "BLACK", "CALLED", "CONCERNS",
            "CRISIS", "DEGREES", "DIFFERENT", "DOING", "EDITOR", "ELECTION",
            "EVERY", "FOCUS", "GROUP", "HUMAN", "IMPACT", "JUSTICE"]
print(len(train_words))


# In[7]:


def video_to_npy_array(video):
    cap = cv2.VideoCapture(video)

    count = 0
    if debug:
        print("Reading file: " + video.split("\\")[-1])

    # initialize MedianFlow tracker for tracking mouth region
    medianflow_tracker = cv2.TrackerMedianFlow_create()

    lip_frames = []
    frames = []
    found_first_frame = False
    found_lips = False
    found_face = False
    video_array = None

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            if len(lip_frames) > 0:
                video_array = np.array(lip_frames, dtype="uint8")
            break
        # convert frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not found_lips:
            # use Haar classifier to find the frontal face
            if debug:
                print("Frontal face located")

            faces = face_haar_cascade.detectMultiScale(gray, 1.2, 3, minSize=(100, 100))
            if len(faces) == 0:
                if debug:
                    print("No Frontal face located")
                    print("Profile face located")
                    
                # if frontal face is not found then try to detect profile face
                faces = face_haar_profile_cascade.detectMultiScale(gray, 1.2, 3, minSize=(100, 100))
                if len(faces) == 0:
                    found_face = False
                    if not found_lips:
                        if debug:
                            print("No Profile face not located --> video skip")
                        break
                else:
                    found_face = True
            else:
                found_face = True

            if found_face:
                face = faces[0]
                face[3] += 20
                face = (x, y, w, h)
                # drawing rectangle for face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                lower_face = int(h * 0.5)
                lower_face_roi = gray[y + lower_face:y + h, x:x + w]

                # detect mouth region in lower half of the face
                mouths = mouth_haar_cascade.detectMultiScale(lower_face_roi, 1.3, 15)
                if len(mouths) > 0:
                    # if first mouth is found
                    mouth = mouths[0]
                    mouth[0] += x  # add face x to fix absolute pos
                    mouth[1] += y + lower_face  # add face y to fix absolute pos

                    m = mouth
                    # drawing rectangle for mouth
                    cv2.rectangle(frame, (m[0], m[1]), (m[0] + m[2], m[1] + m[3]), (0, 255, 0), 2)

                    # initialized the init tracker
                    if not found_lips:
                        lip_track = mouth
                        # extend tracking area
                        lip_track[0] -= 10
                        lip_track[1] -= 20
                        lip_track[2] += 20
                        lip_track[3] += 30
                        medianflow_tracker.init(frame, tuple(lip_track))
                        found_lips = True

                    if count == 0:
                        found_first_frame = True

                    if not found_first_frame:
                        cap = cv2.VideoCapture(video)
                        found_first_frame = True
                        continue

                # skip the sample, if the face is not found
                else:
                    if not found_lips:
                        if debug:
                            print("Lips not found, skipping video")
                        break
        # Update medianflow tracker
        else:
            ok, bbox = medianflow_tracker.update(frame)
            # if tracker is successfully matched in following frame
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)

                lips_roi = gray[
                           int(bbox[1]):int(bbox[1]) + int(bbox[3]),
                           int(bbox[0]):int(bbox[0]) + int(bbox[2])
                           ]

                # prevent crash when tracker goes out of frame
                # and skip video if this occurs (eg. waved hand in front of mouth...)
                if lips_roi.size == 0:
                    break

                lips_resized = cv2.resize(lips_roi, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
                lip_frames.append(lips_resized)
                if debug:
                    cv2.imwrite('outputs/sample/frame-' + str(count) + ".png", lips_resized)

            # if tracker is lost, skip the sample
            else:
                if debug:
                    print("lost tracker")
                break
        if debug:
            if len(frames) != DEPTH:
                cv2.imwrite('outputs/haar/frame-' + str(count) + ".png", frame)
                frames.append(frame)
        count += 1
        if count > DEPTH:
            video_array = np.array(lip_frames, dtype="uint8")
            break

    cap.release()
    return video_array


# In[8]:


# crop and store the preprocessed images
def crop_store_data(set_type, words):

    # if NPY_FOLDER not found, then create the folder
    if not os.path.exists(NPY_FOLDER):
        os.makedirs(NPY_FOLDER)

    # if TYPE folder not found, then create the folder
    if not os.path.exists(os.path.join(NPY_FOLDER, set_type)):
        os.makedirs(os.path.join(NPY_FOLDER, set_type))

    for current_word in words:

        word_data = glob.glob(os.path.join(video_path, current_word, set_type, '') + "*.mp4")
        stored = 0
        not_stored = 0

        # cycle thru word samples
        for i, video in enumerate(word_data):
            npy = video_to_npy_array(video)

            if npy is not None:
                if not os.path.exists(os.path.join(NPY_FOLDER, set_type, current_word)):
                    os.makedirs(os.path.join(NPY_FOLDER, set_type, current_word))
                np.save(os.path.join(NPY_FOLDER, set_type, current_word, str(i)), npy)
                stored += 1
            else:
                not_stored += 1

        print("Word: {}, stored/not stored: {}/{}".format(current_word, stored, not_stored))


# Save Pre-processed Data

# In[ ]:


crop_store_data("train", train_words)
crop_store_data("val", train_words)
crop_store_data("test", train_words)

