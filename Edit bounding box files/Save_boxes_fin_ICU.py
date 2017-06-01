
# coding: utf-8

# In[ ]:

import os,sys
import glob
import cv2
import PIL
import numpy as np
import pandas as pd
from skimage import io

#import matplotlib
#matplotlib.use('Agg')

#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import toimage
from scipy.signal import medfilt
from scipy.ndimage.filters import median_filter
#get_ipython().magic(u'matplotlib inline')
import shutil as sh


if __name__ == "__main__":
    # List of all files with median filtered bounding box coordinates
    tmpf = '/data0/madhu/Dataset/Training/Bounds_ICU_fin/'
    tmpi = '/data0/madhu/Dataset/OR_data/Faces_ICU_fin/'
    tmpi2 = '/data0/madhu/Dataset/OR_data/Boxes_ICU_fin/'
    dirs_bounds = sorted(os.walk(tmpf).next()[2])
    # List of all folders with images
    dirs_images = sorted(os.walk(tmpi).next()[1])
    dirs_boxes = sorted(os.walk(tmpi2).next()[1])
    
    
    #Iterate across text files and images
    for i_fil in range(np.size(dirs_images)):
        
        #Output diectory to store the box images
        directory = os.path.dirname(tmpi2+dirs_images[i_fil]+'/')
        #Clear contents of folder
        if os.path.exists(directory):
            sh.rmtree(directory)
        #Create directory if not present
        if not os.path.exists(directory):
            os.makedirs(directory)

        for ffilename in sorted(glob.glob(tmpf+dirs_images[i_fil])): #assuming jpg
            print (dirs_images[i_fil])
            #if dirs_bounds[i_fil]=='GOPR0854':
            #continue
            #x='data'+str(i_fil)   
            #vars()[x] = np.loadtxt(ffilename)
            data = np.loadtxt(ffilename)
            
            ctr = -1
            ctr2 = 0
                
            for ifilename in sorted(glob.glob(tmpi+dirs_images[i_fil]+'/*.jpg')): #assuming jpg
                #print (ifilename)
                ctr = ctr+1
                ctr2 = ctr2+1
                #if dirs_bounds[i_fil] == 'GOPR0863':
                if ctr2 == 400:
                    print (ctr)
                    ctr2 = 0
                    #if ctr<=20150:
                        #continue
                    #if ctr>=20170:
                        #raw_input("Press Enter to continue...")
                imm = cv2.imread(ifilename)
                #print (imm)
                #h = vars()[x]
                left = int(data[ctr, 0])
                top = int(data[ctr, 1])
                right = int(data[ctr, 2])
                bottom = int(data[ctr, 3])
                cv2.rectangle(imm,(left,top),(right,bottom),(0,255,0),3)
                file_path = tmpi2+dirs_images[i_fil]+'/box'+str(ctr).zfill(8)+'.jpg'
                #plt.imshow(imm)
                #plt.savefig(file_path)
                #plt.close()
		cv2.imwrite(file_path, imm)
                #Image.open(file_path3).save('testplot.jpg','JPEG')


