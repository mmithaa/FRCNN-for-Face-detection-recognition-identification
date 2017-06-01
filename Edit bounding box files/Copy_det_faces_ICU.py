import os,sys
import glob
import cv2
import PIL
import numpy as np
import pandas as pd
from skimage import io

#import matplotlib
#matplotlib.use('Agg')

#%matplotlib inline  
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import toimage
from scipy.signal import medfilt
from scipy.ndimage.filters import median_filter
#%matplotlib inline 
import shutil as sh

faces_file = '/data0/madhu/Dataset/OR_data/Images/'
bounds_file = '/data0/madhu/Dataset/Training/Bounds_ICU/'
faces_out = '/data0/madhu/Dataset/OR_data/Faces_ICU_fin/'
bounds_out = '/data0/madhu/Dataset/Training/Bounds_ICU_fin/'
dirs_faces = sorted(os.walk(faces_file).next()[1])
dirs_bounds = sorted(os.walk(bounds_file).next()[2])

#Output diectory to store FINAL FACES
directory = os.path.dirname(faces_out)
#print ('directory=',directory)
#Delete folder to clear its contents
if os.path.exists(directory):
    sh.rmtree(directory)
#Create directory if not present
if not os.path.exists(directory):
    os.makedirs(directory)

#Output diectory to store dir
directory = os.path.dirname(bounds_out)
#print ('directory=',directory)
#Delete folder to clear its contents
if os.path.exists(directory):
    sh.rmtree(directory)
#Create directory if not present
if not os.path.exists(directory):
    os.makedirs(directory)


ctr = -1
copy_array2 = []
#Iterate across bounds files (counter)
for i_fil in range(np.size(dirs_bounds)):

    #Output diectory to store FINAL FACES
    directory = os.path.dirname(faces_out+dirs_bounds[i_fil]+'/')
    #print ('directory=',directory)
    #Delete folder to clear its contents
    if os.path.exists(directory):
    	sh.rmtree(directory)
    #Create directory if not present
    if not os.path.exists(directory):
	os.makedirs(directory)

    #Iterate across bounds files (actual)
    for ffilename in sorted(glob.glob(bounds_file+dirs_bounds[i_fil])):
	copy_array = []
        print (ffilename)
        del_array = []
        dataa = np.loadtxt(ffilename)

        #Check which image i.e., row has all 0 bounds and append the imdex to a numpy array
        for i in range (np.shape(dataa)[0]):
            if (dataa[i]).all() == np.array([0, 0, 0, 0]).all():
                del_array.append(i)
            else:
            #Copy corresponding values to a new text file
                ctr = ctr+1
                copy_array2.append(dataa[i])
		copy_array.append(dataa[i])
                   
                
                #Iterate across images in a folder
                #for ifilename in sorted(glob.glob(faces_file+dirs_bounds[i_fil]+'/*.jpg')): #assuming jpg
                img = cv2.imread(faces_file+dirs_bounds[i_fil]+'/out-'+str(i+1).zfill(6)+'.jpg')
                cv2.imwrite(faces_out+dirs_bounds[i_fil]+'/'+str(ctr).zfill(7)+'.jpg', img)
		print (faces_out+dirs_bounds[i_fil]+'/'+str(ctr).zfill(7)+'.jpg')

    	#Write non-zero bounds to a file
    	np.savetxt(bounds_out+dirs_bounds[i_fil], copy_array) 
            
#Write non-zero bounds to a file
np.savetxt(bounds_out+'Bounds_fin.txt', copy_array2)           
