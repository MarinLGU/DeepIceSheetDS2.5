import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import netCDF4 as nc
import h5py
from utils import make_data


ds=nc.Dataset('data/RCP_start.nc')
thk=ds.variables['thk'][0,314:724,210:1030]
thk=thk/4693.41
topg=ds.variables['topg'][0,314:724,210:1030]
topg=topg/topg.max()
x=ds.variables['x'][210:1030]
y=ds.variables['y'][314:724]
X,Y=np.meshgrid(x,y)
thkHR=np.zeros((200,41,41,4))
for i in range(10):
    for j in range(20):
        thkHR[i*20+j,:,:,0]=thk[i*41:(i+1)*41,j*41:(j+1)*41] #ice thk as channel 0
        thkHR[i*20+j,:,:,1]=topg[i*41:(i+1)*41,j*41:(j+1)*41] #topography as channel 1
        thkHR[i*20+j,:,:,2]=X[i*41:(i+1)*41,j*41:(j+1)*41]  #X grid as channel 2
        thkHR[i*20+j,:,:,3]=Y[i*41:(i+1)*41,j*41:(j+1)*41] #Y grid as channel 3



import scipy.ndimage

thkLR=scipy.ndimage.zoom(thkHR,(1,0.25,0.25,1),order=3) #bicubic upscaling
thkLR=scipy.ndimage.zoom(thkLR,(1,4.1,4.1,1),order=3) #bicubic downscaling

###Refilling thkLR with HR topg, X & Y grids
for i in range(10):
    for j in range(20):
        thkLR[i*20+j,:,:,1]=topg[i*41:(i+1)*41,j*41:(j+1)*41] #topography as channel 1
        thkLR[i*20+j,:,:,2]=X[i*41:(i+1)*41,j*41:(j+1)*41]  #X grid as channel 2
        thkLR[i*20+j,:,:,3]=Y[i*41:(i+1)*41,j*41:(j+1)*41] #Y grid as channel 3




from sklearn.model_selection import train_test_split

thkLR_Train,thkLR_Test, thkHR_Train, thkHR_Test=train_test_split(thkLR,thkHR,test_size=0.3)


###Uncomment only if you want to create new training/testing set. Be aware, it'll not be relevant anymore to test model already trained on new testing set###



# h5f= h5py.File("train_x4", "w")
# h5f.create_dataset("data", data=thkLR_Train)
# h5f.create_dataset("label", data=thkHR_Train)
# h5f.close()
#
# h5f= h5py.File("test_x4", "w")
# h5f.create_dataset("data", data=thkLR_Test)
# h5f.create_dataset("label", data=thkHR_Test)
# h5f.close()
#
# import tensorflow as tf
# trainset=tf.data.Dataset.from_tensor_slices((thkLR_Train,thkHR_Train))
# testset=tf.data.Dataset.from_tensor_slices((thkLR_Test,thkHR_Test))





