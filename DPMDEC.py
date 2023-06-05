from dp_init import *
from vdpmm_maximizePlusGaussian import *
from vdpmm_expectationPlusGaussian import *
from SVDOPPos import *
def DPMDEC(maps,ep):
    numits=1;
    maxits=20;
    K=20;
    maps=maps
    paramsGaussian,posGaussian = vdpmm_init(maps,K)
    for i in range(maxits):
        paramsGaussian = vdpmm_maximizePlusGaussian(maps, paramsGaussian, posGaussian)
        posGaussian = vdpmm_expectationPlusGaussian(maps, paramsGaussian)
        if i%3==0:
            posGaussian=SVDOPPos(posGaussian,ep)
        #print(i)
        # posGaussian=vdpmm_expectationCNNKnearset(data,params,0.4,posGaussian)
        # params=vdpmm_maximizeCNN(data,params,posGaussian)
    [Nz, Dz] = maps.shape
    temp = np.max(posGaussian, axis=1)
    temp.shape = (Nz, 1)
    index1 = np.where(temp == posGaussian)
    return index1[1]