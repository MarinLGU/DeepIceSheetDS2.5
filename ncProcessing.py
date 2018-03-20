import numpy as np
import scipy.ndimage as sci
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import netCDF4 as nc
import h5py
from utils import make_data
from main import FLAGS



ds=nc.Dataset('data/RCP_start.nc')
thk=ds.variables['thk'][0,314:724,210:1030]
thk=thk/thk.max()
topg=ds.variables['topg'][0,314:724,210:1030]
topg=topg/topg.max()
x=ds.variables['x'][210:1030]
y=ds.variables['y'][314:724]
X,Y=np.meshgrid(x,y)

def make_datasets(config):
    thkHR=np.zeros((200,config.label_size,config.label_size,4))
    for i in range(10):
        for j in range(20):
            thkHR[i*20+j,:,:,0]=thk[i*41:(i+1)*41,j*41:(j+1)*41] #ice thk as channel 0
            thkHR[i*20+j,:,:,1]=topg[i*41:(i+1)*41,j*41:(j+1)*41] #topography as channel 1
            thkHR[i*20+j,:,:,2]=X[i*41:(i+1)*41,j*41:(j+1)*41]  #X grid as channel 2
            thkHR[i*20+j,:,:,3]=Y[i*41:(i+1)*41,j*41:(j+1)*41] #Y grid as channel 3


    thkLR=np.zeros((200,config.image_size,config.image_size,4))
    s1=thkLR.shape
    temp=sci.zoom(thkHR[:,:,:,0],(1,1/config.scale_factor,1/config.scale_factor))
    s2=temp.shape
    thkLR[:, :, :, 0] = sci.zoom(temp,(1,s1[1]/s2[1],s1[2]/s2[2])) #ice thk upscaled as channel 0
    ###Refilling thkLR with HR topg, X & Y grids
    for i in range(10):
        for j in range(20):
            thkLR[i*20+j,:,:,1]=topg[i*41:(i+1)*41,j*41:(j+1)*41] #topography as channel 1
            thkLR[i*20+j,:,:,2]=X[i*41:(i+1)*41,j*41:(j+1)*41]  #X grid as channel 2
            thkLR[i*20+j,:,:,3]=Y[i*41:(i+1)*41,j*41:(j+1)*41] #Y grid as channel 3




    from sklearn.model_selection import train_test_split

    thkLR_Train,thkLR_Test, thkHR_Train, thkHR_Test=train_test_split(thkLR,thkHR,test_size=0.3)

    if config.save_datasets :

        h5f= h5py.File("train_x%i" %config.scale_factor, "w")
        h5f.create_dataset("data", data=thkLR_Train)
        h5f.create_dataset("label", data=thkHR_Train)
        h5f.close()

        h5f= h5py.File("test_x%i" %config.scale_factor, "w")
        h5f.create_dataset("data", data=thkLR_Test)
        h5f.create_dataset("label", data=thkHR_Test)
        h5f.close()

    return thkLR_Train,thkLR_Test, thkHR_Train, thkHR_Test

make_datasets(FLAGS)





