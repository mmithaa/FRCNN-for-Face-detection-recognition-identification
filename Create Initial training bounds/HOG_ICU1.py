
# coding: utf-8

# # Face Detection in training images using HOG classifier

# In[ ]:

# Import deep learning, numeric, dataframe and opencv libraries
import os,sys
import glob
import dlib
import cv2
import PIL
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import toimage
#get_ipython().magic(u'matplotlib inline')


# # Haar cascades

# In[ ]:

def classifier(frame):
    face_cascade = cv2.CascadeClassifier('/home/madhumitha/Madhu/Code/OpenCV/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
    #frame = cv2.imread(tmp)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    # Display the resulting frame
    #cv2.imshow('frame', frame)
    return [frame, roi_color, x, y, w, h]

    


# # HOG

# In[ ]:

def classifier2(filename):
    # Apply HOG and perform face detection, save coordinates of face to a text file
    detector = dlib.get_frontal_face_detector()
    #win = dlib.image_window()

    #for f in sys.argv[1:]:
    #print("Processing file: {}".format(f))
    img = io.imread(filename)
    #img = np.asarray(jpgfile)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces
    dets1 = detector(img, 0)
    #print("Number of faces detected: {}".format(len(dets1)))
    #for i, d in enumerate(dets1):
    #    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #        i, d.left(), d.top(), d.right(), d.bottom()))

    #win.clear_overlay()
    #win.set_image(img)
    #win.add_overlay(dets)
    #dlib.hit_enter_to_continue()


    # Finally, if you really want to you can ask the detector to tell you the score
    # for each detection.  The score is bigger for more confident detections.
    # The third argument to run is an optional adjustment to the detection threshold,
    # where a negative value will return more detections and a positive value fewer.
    # Also, the idx tells you which of the face sub-detectors matched.  This can be
    # used to broadly identify faces in different orientations.

    #if (len(sys.argv[1:]) > 0):
        #img = io.imread(sys.argv[1])
    #dets, scores, idx = detector.run(img, 1, -1)
    #for i, d in enumerate(dets):
    #    print(str(i))
    #    print("Detection {}, score: {}, face_type:{}".format(
    #        d, scores[i], idx[i]))
    return [dets1]


# In[ ]:

# Import input images for face detetcion using HOG
tmp = '/data0/madhu/Dataset/OR_data/Images/ICU_TEST1/out-000014.jpg'
tmp3 = '/data0/madhu/Dataset/OR_data/Images/'
tmp2 = '/data0/madhu/Dataset/Training/Bounds_ICU/'

#jpgfile = Image.open(tmp1)
#print jpgfile.bits, jpgfile.size, jpgfile.format
#jpgfile

# In[ ]:

allimage_list = []
dirs = sorted(os.walk(tmp3).next()[1])

for i in range(np.size(dirs)):
    text_array = []
    print (dirs[i])

    for filename in sorted(glob.glob(tmp3+dirs[i]+'/*.jpg')): #assuming jpg
        #imm = cv2.imread(filename)
        #[frame, roi_color, x, y, w, h] = classifier (imm)
        [dets1] = classifier2(filename) 
        #np.savetxt(tmp2, data[None, :], delimiter = ' ')
        #f_handle = open(tmp2+dirs[i],"w+")
        #f_handle.close()

        if np.size(dets1):
            data = np.array([dets1[0].left(), dets1[0].top(), dets1[0].right(), dets1[0].bottom()])
            #f_handle = file(tmp2+dirs[i], 'a')
            #np.savetxt(f_handle, data[None, :], delimiter = ' ')
            text_array.append(data)
            print ([dets1[0].left(), dets1[0].top(), dets1[0].right(), dets1[0].bottom()])
            #f_handle.close()
        else:
            dummy = np.array([0, 0, 0, 0])
            text_array.append(dummy)
            #f_handle = file(tmp2+dirs[i], 'a')
            #np.savetxt(f_handle, dummy[None, :], delimiter = ' ')
            #f_handle.close()
            
    np.savetxt(tmp2+dirs[i], text_array)

        
        #allimage_list.append(imm)
    
    
        



