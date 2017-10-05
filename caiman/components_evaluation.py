# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:12:34 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
import numpy as np
from .utils.stats import mode_robust, mode_robust_fast
from scipy.sparse import csc_matrix
from scipy.stats import norm
import scipy
import cv2
try:
	import json as simplejson
	from keras.models import model_from_json
except:
	print('KERAS NOT INSTALLED. IF YOU WANT TO USE THE CNN BASED COMPONENT CLASSIFIER (experimental) CONTACT THE DEVELOPERS')

def estimate_noise_mode(traces,robust_std=False,use_mode_fast=False, return_all = False):
    """ estimate the noise in the traces under assumption that signals are sparse and only positive. The last dimension should be time. 

    """
    # todo todocument
    if use_mode_fast:
        md = mode_robust_fast(traces, axis=1)
    else:
        md = mode_robust(traces, axis=1)

    ff1 = traces - md[:, None]
    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:
        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, el in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(old_div(np.sum(ff1**2, 1), Ns))
    
    if return_all :
        return md, sd_r    
    else:
        return sd_r    
#


#%%
try:
    profile
except:
    profile = lambda a: a 


@profile
def compute_event_exceptionality(traces,robust_std=False,N=5,use_mode_fast=False):
    """
    Define a metric and order components according to the probabilty if some "exceptional events" (like a spike). 

    Such probability is defined as the likeihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode. 
    This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349. 
    Then, the probavility of having N consecutive eventsis estimated.
    This probability is used to order the components.

    Parameters:    
    -----------
    Y: ndarray 
        movie x,y,t

    A: scipy sparse array
        spatial components    

    traces: ndarray
        Fluorescence traces 

    N: int
        N number of consecutive events

    Returns:
    --------

    fitness: ndarray
        value estimate of the quality of components (the lesser the better)

    erfc: ndarray
        probability at each time step of observing the N consequtive actual trace values given the distribution of noise

    noise_est: ndarray
        the components ordered according to the fitness

    """
    T=np.shape(traces)[-1]
    if use_mode_fast:
        md = mode_robust_fast(traces, axis=1)
    else:
        md = mode_robust(traces, axis=1)

    ff1 = traces - md[:, None]

    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:

        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, el in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(old_div(np.sum(ff1**2, 1), Ns))

    # compute z value
    z = old_div((traces - md[:, None]), (3 * sd_r[:, None]))

    # probability of observing values larger or equal to z given normal
    # distribution with mean md and std sd_r
    erf = 1 - norm.cdf(z)

    # use logarithm so that multiplication becomes sum
    erf = np.log(erf)
    filt = np.ones(N)

    # moving sum
    erfc = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='full'), axis=1, arr=erf)
    erfc = erfc [:,:T]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    return fitness,erfc,sd_r,md


#%%
def find_activity_intervals(C,Npeaks = 5, tB=-3, tA = 10, thres = 0.3):
#todo todocument
    import peakutils
    K,T = np.shape(C)
    L = []
    for i in range(K):
        if np.sum(np.abs(np.diff(C[i,:])))==0:
            L.append([])        
            print('empyty component at:'+str(i))
            continue
        indexes = peakutils.indexes(C[i,:],thres=thres)        
        srt_ind = indexes[np.argsort(C[i,indexes])][::-1]
        srt_ind = srt_ind[:Npeaks]
        L.append(srt_ind)

    LOC = []
    for i in range(K):
        if len(L[i])>0:
            interval = np.kron(L[i],np.ones(int(np.round(tA-tB)),dtype=int)) + np.kron(np.ones(len(L[i]),dtype=int),np.arange(tB,tA))                        
            interval[interval<0] = 0
            interval[interval>T-1] = T-1
            LOC.append(np.array(list(set(interval))))        
        else:
            LOC.append(None)

    return LOC


#%%
def classify_components_ep(Y,A,C,b,f,Athresh = 0.1,Npeaks = 5, tB=-3, tA = 10, thres = 0.3):
    # todo todocument

    K,T = np.shape(C)
    A = csc_matrix(A)
    AA = (A.T*A).toarray() 
    nA=np.sqrt(np.array(A.power(2).sum(0)))
    AA = old_div(AA,np.outer(nA,nA.T))
    AA -= np.eye(K)
    
    LOC = find_activity_intervals(C, Npeaks = Npeaks, tB=tB, tA = tA, thres = thres)
    rval = np.zeros(K)

    significant_samples=[]
    for i in range(K):
        if i%200 == 0:
            print('components evaluated:'+str(i))
        if LOC[i] is not None:
            atemp = A[:,i].toarray().flatten()
            atemp[np.isnan(atemp)] = np.nanmean(atemp)
            ovlp_cmp = np.where(AA[:,i]>Athresh)[0]
            indexes = set(LOC[i])
            for cnt,j in enumerate(ovlp_cmp):
                if LOC[j] is not None:
                    indexes = indexes - set(LOC[j])
            
            if len(indexes) == 0:
                indexes = set(LOC[i])
                print('Neuron:' + str(i) + ' includes overlaping spiking neurons')
                
            indexes = np.array(list(indexes)).astype(np.int)
            px = np.where(atemp>0)[0]
            ysqr = np.array(Y[px,:])
            ysqr[np.isnan(ysqr)] = np.nanmean(ysqr) 
            mY = np.mean(ysqr[:,indexes],axis=-1)
            significant_samples.append(indexes)
            rval[i] = scipy.stats.pearsonr(mY,atemp[px])[0]

        else:            
            rval[i] = 0
            significant_samples.append(0)

    return rval,significant_samples
#%%
def evaluate_components_CNN(A,dims,gSig,model_name = 'use_cases/CaImAnpaper/cnn_model', patch_size = 50):
    """ evaluate component quality using a CNN network
    
    """
    json_file = open(model_name +'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name +'.h5')
    print("Loaded model from disk")
    half_crop = np.minimum(gSig[0]*4+1,patch_size)
    dims = np.array(dims)
    coms = [scipy.ndimage.center_of_mass(mm.toarray().reshape(dims,order='F')) for mm in A.tocsc().T]    
    coms = np.maximum(coms,half_crop)
    coms = np.array([np.minimum(cms,dims-half_crop) for cms in coms]).astype(np.int)
    crop_imgs = [mm.toarray().reshape(dims,order='F')[com[0]-half_crop:com[0]+half_crop, com[1]-half_crop:com[1]+half_crop] for mm,com in zip(A.tocsc().T,coms) ]
    final_crops = np.array([cv2.resize(im/np.linalg.norm(im),(patch_size ,patch_size )) for im in crop_imgs])
    predictions = loaded_model.predict(final_crops[:,:,:,np.newaxis], batch_size=32, verbose=1)

    return predictions,final_crops
#%%
def evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline = True, N = 5, robust_std = False,
                        Athresh = 0.1, Npeaks = 5, thresh_C = 0.3):
    """ Define a metric and order components according to the probabilty if some "exceptional events" (like a spike).
    
    Such probability is defined as the likeihood of observing the actual trace value over N samples given an estimated noise distribution. 
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode.
    The estimation of the noise std is made robust by using the approximation std=iqr/1.349. 
    Then, the probavility of having N consecutive eventsis estimated.
    This probability is used to order the components. 
    The algorithm also measures the reliability of the spatial mask by comparing the filters in A
     with the average of the movies over samples where exceptional events happen, after  removing (if possible)
    frames when neighboring neurons were active

    Parameters:
    ----------
    Y: ndarray 
        movie x,y,t

    A,C,b,f: various types 
        outputs of cnmf    

    traces: ndarray
        Fluorescence traces 

    remove_baseline: bool
        whether to remove the baseline in a rolling fashion *(8 percentile)

    N: int
        N number of consecutive events probability multiplied


    Athresh: float 
        threshold on overlap of A (between 0 and 1)

    Npeaks: int

   

    thresh_C: float
        fraction of the maximum of C that is used as minimum peak height        

    Returns:
    -------
    idx_components: ndarray
        the components ordered according to the fitness

    fitness_raw: ndarray
        value estimate of the quality of components (the lesser the better) on the raw trace

    fitness_delta: ndarray
        value estimate of the quality of components (the lesser the better) on diff(trace)

    erfc_raw: ndarray
        probability at each time step of observing the N consequtive actual trace values given the distribution of noise on the raw trace

    erfc_raw: ndarray
        probability at each time step of observing the N consequtive actual trace values given the distribution of noise on diff(trace)    

    r_values: list
        float values representing correlation between component and spatial mask obtained by averaging important points

    significant_samples: ndarray
        indexes of samples used to obtain the spatial mask by average

    """
    tB = np.minimum(-2, np.floor( -5. / 30 * final_frate))
    tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
    print('tB:'+str(tB)+',tA:'+str(tA))
    dims,T=np.shape(Y)[:-1],np.shape(Y)[-1]
    
    Yr=np.reshape(Y,(np.prod(dims),T),order='F')    

    print('Computing event exceptionality delta')
    fitness_delta, erfc_delta,std_rr, _ = compute_event_exceptionality(np.diff(traces,axis=1),robust_std=robust_std,N=N)

    print('Removing Baseline')
    if remove_baseline:
        num_samps_bl=np.minimum(old_div(np.shape(traces)[-1],5),800)
        slow_baseline = False
        if slow_baseline:
            
            traces = traces - scipy.ndimage.percentile_filter(traces,8,size=[1,num_samps_bl])

        else: # fast baseline removal
            
            downsampfact = num_samps_bl
            elm_missing=int(np.ceil(T*1.0/downsampfact)*downsampfact-T)
            padbefore=int(np.floor(old_div(elm_missing,2.0)))
            padafter=int(np.ceil(old_div(elm_missing,2.0)))    
            tr_tmp = np.pad(traces.T,((padbefore,padafter),(0,0)),mode='reflect')
            numFramesNew,num_traces = np.shape(tr_tmp)    
            #% compute baseline quickly
            print("binning data ..."); 
            tr_BL=np.reshape(tr_tmp,(downsampfact,int(old_div(numFramesNew,downsampfact)),num_traces),order='F');
            tr_BL=np.percentile(tr_BL,8,axis=0)            
            print("interpolating data ..."); 
            print(tr_BL.shape)    
            tr_BL=scipy.ndimage.zoom(np.array(tr_BL,dtype=np.float32),[downsampfact ,1],order=3, mode='constant', cval=0.0, prefilter=True)
            if padafter==0:
                traces -= tr_BL.T
            else:
                traces -= tr_BL[padbefore:-padafter].T
            
    print('Computing event exceptionality')    
    fitness_raw, erfc_raw,std_rr, _ = compute_event_exceptionality(traces,robust_std=robust_std,N=N)

    print('Evaluating spatial footprint')
    # compute the overlap between spatial and movie average across samples with significant events
    r_values, significant_samples = classify_components_ep(Yr, A, C, b, f, Athresh = Athresh, Npeaks = Npeaks, tB=tB,
                                                           tA = tA, thres = thresh_C)

    return fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples    


#%%
def chunker(seq, size):
    for pos in xrange(0, len(seq), size):
        yield seq[pos:pos + size]
#%%
def estimate_components_quality(traces, Y, A, C, b, f, final_frate = 30, Npeaks=10, r_values_min = .95,
                                fitness_min = -100,fitness_delta_min = -100, return_all = False, N =5, remove_baseline = True):
    """ Define a metric and order components according to the probabilty if some "exceptional events" (like a spike).

    Such probability is defined as the likeihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode.
    The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probavility of having N consecutive eventsis estimated.
    This probability is used to order the components.
    The algorithm also measures the reliability of the spatial mask by comparing the filters in A
     with the average of the movies over samples where exceptional events happen, after  removing (if possible)
    frames when neighboring neurons were active

    Parameters:
    ----------
    Y: ndarray
        movie x,y,t

    A,C,b,f: various types
        outputs of cnmf

    traces: ndarray
        Fluorescence traces

    N: int
        N number of consecutive events probability multiplied

    Npeaks: int

    r_values_min: list
        minimum correlation between component and spatial mask obtained by averaging important points

    fitness_min: ndarray
        minimum acceptable quality of components (the lesser the better) on the raw trace

    fitness_delta_min: ndarray
        minimum acceptable the quality of components (the lesser the better) on diff(trace)

    thresh_C: float
        fraction of the maximum of C that is used as minimum peak height

    Returns:
    -------
    idx_components: ndarray
        the components ordered according to the fitness

    idx_components_bad: ndarray
        the components ordered according to the fitness

    fitness_raw: ndarray
        value estimate of the quality of components (the lesser the better) on the raw trace

    fitness_delta: ndarray
        value estimate of the quality of components (the lesser the better) on diff(trace)

    r_values: list
        float values representing correlation between component and spatial mask obtained by averaging important points

    """

    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
        evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=remove_baseline,
                                          N=N, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3 )
    
    idx_components_r = np.where(r_values >= r_values_min)[0]  # threshold on space consistency
    idx_components_raw = np.where(fitness_raw < fitness_min)[0] # threshold on time variability
    idx_components_delta = np.where(fitness_delta < fitness_delta_min)[0] # threshold on time variability (if nonsparse activity)
    
    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)
    
    if return_all:
        return idx_components,idx_components_bad, fitness_raw, fitness_delta, r_values 
    else:
        return idx_components,idx_components_bad
