# -*- coding: utf-8 -*-
"""
OPTICAL FLOW

Created on Wed Mar 16 16:31:55 2016

@author: agiovann
"""

from __future__ import division
from __future__ import print_function
#%%
from builtins import zip
from builtins import range
from past.utils import old_div
import caiman as cm
import numpy as np
import pylab as pl
from scipy.io import loadmat
import cv2
from sklearn.decomposition import NMF,PCA,DictionaryLearning
import time
import scipy
from scipy.sparse import coo_matrix


#%% dense flow
def select_roi(img,n_rois=1):
    """
    Create a mask from a the convex polygon enclosed between selected points

    Parameters:
    ----------
    img: 2D ndarray
        image used to select the points for the mask

    n_rois: int
        number of rois to select

    Returns:
    -------
    mask: list
        each element is an the mask considered a ROIs
    """

    masks=[]
    for _ in range(n_rois):
        fig=pl.figure()
        pl.imshow(img,cmap=pl.cm.gray)
        pts = fig.ginput(0, timeout=0)
        mask = np.zeros(np.shape(img), dtype=np.int32)
        pts = np.asarray(pts, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, (1,1,1), lineType=cv2.LINE_AA)
        masks.append(mask)
        pl.close()

    return masks


#%%
def to_polar(x,y):
    mag, ang = cv2.cartToPolar(x, y)
    return mag,ang

#%%
def get_nonzero_subarray(arr,mask):
    x, y = mask.nonzero()
    return arr.toarray()[x.min():x.max()+1, y.min():y.max()+1]
    
#%%
def extract_motor_components_OF(m, n_components, mask = None,  resize_fact= .5, only_magnitude = False, max_iter = 1000,
                                verbose = False, method_factorization = 'nmf',max_iter_DL=-30):
        #todo todocument
    if mask is not None:
        mask = coo_matrix(np.array(mask).squeeze())
        ms = [get_nonzero_subarray(mask.multiply(fr),mask) for fr in m]
        ms = np.dstack(ms)
        ms = cm.movie(ms.transpose([2,0,1]))

    else:
        ms = m
    of_or = compute_optical_flow(ms,do_show=False,polar_coord=False) 
    of_or = np.concatenate([cm.movie(of_or[0]).resize(resize_fact,resize_fact,1)[np.newaxis,:,:,:],
                            cm.movie(of_or[1]).resize(resize_fact,resize_fact,1)[np.newaxis,:,:,:]],axis = 0)

    if only_magnitude:
        of = np.sqrt(of[0]**2+of[1]**2)
    else: 
        if method_factorization == 'nmf':
            offset_removed = np.min(of_or)        
            of = of_or - offset_removed        
        else:
            of = of_or 
            
    spatial_filter_, time_trace_, norm_fact = extract_components(of,n_components=n_components,verbose = verbose ,
                                                                 normalize_std=False,max_iter=max_iter,
                                                                 method_factorization = method_factorization,
                                                                 max_iter_DL=max_iter_DL)
  
    return spatial_filter_, time_trace_, of_or


#%%
def extract_magnitude_and_angle_from_OF(spatial_filter_, time_trace_, of_or,
                                        num_std_mag_for_angle = .6, sav_filter_size =3, only_magnitude = False):
    # todo todocument

    mags = []
    dircts = []
    dircts_thresh = []
    n_components = len(spatial_filter_)
    spatial_masks_thr = []
    for ncmp in range(n_components):
        spatial_filter = spatial_filter_[ncmp]
        time_trace = time_trace_[ncmp]
        if only_magnitude:
            mag = scipy.signal.medfilt(time_trace,kernel_size=[1,1]).T
            mag =  scipy.signal.savgol_filter(mag.squeeze(),sav_filter_size,1)                
            dirct = None
        else:
            x,y = scipy.signal.medfilt(time_trace,kernel_size=[1,1]).T
            x =  scipy.signal.savgol_filter(x.squeeze(),sav_filter_size,1)                
            y =  scipy.signal.savgol_filter(y.squeeze(),sav_filter_size,1)
            mag,dirct = to_polar(x-cm.components_evaluation.mode_robust(x),y-cm.components_evaluation.mode_robust(y))
            dirct = scipy.signal.medfilt(dirct.squeeze(),kernel_size=1).T        
            
        # normalize to pixel units
        spatial_mask = spatial_filter
        spatial_mask[spatial_mask<(np.nanpercentile(spatial_mask[0],99.9)*.9)] = np.nan
        ofmk = of_or*spatial_mask[None,None,:,:]
        range_ = np.std(np.nanpercentile(np.sqrt(ofmk[0]**2+ofmk[1]**2),95,(1,2)))
        mag = (mag/np.std(mag))*range_
        mag = scipy.signal.medfilt(mag.squeeze(),kernel_size=1)
        dirct_orig = dirct.copy()
        if not only_magnitude:
            dirct[mag<num_std_mag_for_angle*np.nanstd(mag)] = np.nan

        mags.append(mag)
        dircts.append(dirct_orig)
        dircts_thresh.append(dirct)
        spatial_masks_thr.append(spatial_mask)
    
    return mags, dircts, dircts_thresh, spatial_masks_thr


    
#%%
def compute_optical_flow(m,mask = None,polar_coord=True,do_show=False,do_write=False,file_name=None,gain_of=None,
                         frate=30,pyr_scale=.1,levels=3 , winsize=25,iterations=3,poly_n=7,poly_sigma=1.5):
    """
    This function compute the optical flow of behavioral movies using the opencv cv2.calcOpticalFlowFarneback function 

    Parameters:
    ----------

    m: 3D ndarray:
        input movie

    mask: 2D ndarray
        mask selecting relevant pixels       

    polar_coord: boolean
        wheather to return the coordinate in polar coordinates (or cartesian)

    do_show: bool
        show flow movie

    do_write: bool
        save flow movie

    frate: double
        frame rate saved movie

    parameters_opencv_function: cv2.calcOpticalFlowFarneback
        pyr_scale,levels,winsize,iterations,poly_n,poly_sigma



    Returns:
    --------
    mov_tot: 4D ndarray containing the movies of the two coordinates

    Raise:
    -----
    Exception('You need to provide file name (.avi) when saving video')

    """        
    prvs=np.uint8(m[0])
    frame1=cv2.cvtColor(prvs, cv2.COLOR_GRAY2RGB)

    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    if do_show:
        cv2.namedWindow( "frame2", cv2.WINDOW_NORMAL )

    if mask is not None:
        data = mask.astype(np.int32)
    else:
        data = 1
        
    T,d1,d2=m.shape
    angs=np.zeros(T)
    mags=np.zeros(T)
    mov_tot=np.zeros([2,T,d1,d2])

    if do_write:
        if file_name is not None:
            video = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc('M','J','P','G'),frate,(d2*2,d1),1)
        else:
            raise Exception('You need to provide file name (.avi) when saving video')

    for counter,next_ in enumerate(m):
        if counter%100 == 0:
            print(counter)          
        frame2 = cv2.cvtColor(np.uint8(next_), cv2.COLOR_GRAY2RGB)    
        flow = cv2.calcOpticalFlowFarneback(prvs,next_, None, pyr_scale, levels, winsize, iterations, poly_n,
                                            poly_sigma, 0)

        if polar_coord:    
            coord_1, coord_2 = cv2.cartToPolar(flow[...,0], flow[...,1])
        else:
            coord_1, coord_2 = flow[:,:,0],flow[:,:,1]        

        coord_1*=data
        coord_2*=data    

        if do_show or do_write:
            if polar_coord:
                hsv[...,0] = coord_2*180/np.pi/2
            else:
                hsv[...,0] = cv2.normalize(coord_2,None,0,255,cv2.NORM_MINMAX)    
            if gain_of is None:
                hsv[...,2] = cv2.normalize(coord_1,None,0,255,cv2.NORM_MINMAX)    
            else:
                hsv[...,2] = gain_of*coord_1
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)    
            frame_tot=np.concatenate([rgb,frame2],axis=1)

        if do_write:
            video.write(frame_tot)

        if do_show:
            cv2.imshow('frame2',frame_tot)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        mov_tot[0,counter]=coord_1
        mov_tot[1,counter]=coord_2

        prvs = next_.copy()

    if do_write:
        video.release()

    if do_show:
        cv2.destroyAllWindows()

    return mov_tot



#%% NMF
def extract_components(mov_tot,n_components=6,normalize_std=True,max_iter_DL=-30, method_factorization = 'nmf', **kwargs):
    """
    From optical flow images can extract spatial and temporal components

    Parameters:
    ----------
    mov_tot: ndarray (can be 3 or 4D)
        contains the optical flow values, either in cartesian or polar, either one (3D) or both (4D coordinates)
        the input is generated by the compute_optical_flow function    

    n_components: int
        number of components to look for

    normalize_std: bool
        whether to normalize each oof the optical flow components

    normalize_output_traces: boolean
        whether to normalize the behavioral traces so that they match the units in the movie

    Returns:
    -------
    spatial_filter: ndarray
        set of spatial inferred filters     

    time_trace:ndarray
        set of time components

    norm_fact: ndarray
        used notmalization factors

    """

    if mov_tot.ndim==4:
        if normalize_std:
            norm_fact=np.nanstd(mov_tot,axis=(1,2,3))    
            mov_tot=old_div(mov_tot,norm_fact[:,np.newaxis,np.newaxis,np.newaxis])
        else:
            norm_fact=np.array([ 1.,  1.])
        c,T,d1,d2=np.shape(mov_tot)

    else:
        norm_fact=1
        T,d1,d2=np.shape(mov_tot)
        c=1

    tt=time.time()    
    newm=np.reshape(mov_tot,(c*T,d1*d2)) 

    if method_factorization == 'nmf':
        nmf=NMF(n_components=n_components,**kwargs)

        time_trace=nmf.fit_transform(newm)
        spatial_filter=nmf.components_
        spatial_filter=np.concatenate([np.reshape(sp,(d1,d2))[np.newaxis,:,:] for sp in spatial_filter],axis=0)
        
    elif method_factorization == 'dict_learn':
        
        import spams
        newm = np.asfortranarray(newm,dtype = np.float32)    
        time_trace = spams.trainDL(newm,K=n_components, mode=0, lambda1 = 1, posAlpha = True, iter = max_iter_DL)

        spatial_filter = spams.lasso(newm, D = time_trace,return_reg_path = False, lambda1 = 0.01,
                                     mode = spams.spams_wrap.PENALTY,  pos=True)

        spatial_filter=np.concatenate([np.reshape(sp,(d1,d2))[np.newaxis,:,:] for sp in spatial_filter.toarray()],axis=0)

    time_trace=[np.reshape(ttr,(c,T)).T for ttr in time_trace.T]

    el_t=time.time()-tt
    print(el_t)
    return spatial_filter,time_trace,norm_fact



def plot_components(sp_filt,t_trace):
    # todo: todocument
    pl.figure()
    count=0
    for comp,tr in zip(sp_filt,t_trace):
        count+=1    
        pl.subplot(6,2,count)
        pl.imshow(comp)
        count+=1            
        pl.subplot(6,2,count)
        pl.plot(tr)


#%%
def normalize_components(t_trace,sp_filt,num_std=2):
    """ 
    Normalize the components using the std of the components obtaining using biunary masks
    """
    # todo: todocument
    coor_1=[]
    coor_2=[]  
    new_t_trace=[]
    for t,s in zip(t_trace,sp_filt):
        print((1))
        thr=np.mean(s)+num_std*np.std(s)
        t=t.T
        t1=t[0]/np.std(t[0])*np.std(old_div(np.sum((s>thr)*mov_tot[0],axis=(1,2)),np.sum((s>thr))))
        t2=t[1]/np.std(t[1])*np.std(old_div(np.sum((s>thr)*mov_tot[1],axis=(1,2)),np.sum((s>thr))))
        coor_1.append(t1)
        coor_2.append(t2)
        new_t_trace.append(np.vstack([t1,t2]).T)

    coor_1=np.array(coor_1)    
    coor_2=np.array(coor_2)

    return new_t_trace,coor_1,coor_2


#%%
if __name__ == "__main__":
    # todo: main ??
    main()

#%%    
def main():
    mmat=loadmat('mov_AG051514-01-060914 C.mat')['mov']
    m=cm.movie(mmat.transpose((2,0,1)),fr=120)
    mask=select_roi(m[0])
    if 1:
        mov_tot=compute_optical_flow(m[:3000],mask)
    else:
        mov_tot=compute_optical_flow(m[:3000],mask,polar_coord=False)

    sp_filt,t_trace,norm_fact=extract_components(mov_tot)
    plot_components(sp_filt,t_trace)
    id_comp=1
    pl.plot(old_div(np.sum(np.reshape(sp_filt[id_comp]>1,[d1,d2])*mov_tot[1],axis=(1,2)),np.sum(sp_filt[id_comp]>1)))
    pl.plot(t_trace[id_comp][:,1]) 




##%% single featureflow,. seems not towork
#feature_params = dict( maxCorners = 100,
#                       qualityLevel = 0.3,
#                       minDistance = 7,
#                       blockSize = 7 )
#
#lk_params = dict( winSize  = (15,15),
#                  maxLevel = 2,
#                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
#
#color = np.random.randint(0,255,(100,3))
#old_gray=m[0]
#old_frame = cv2.cvtColor(old_gray, cv2.COLOR_GRAY2RGB)
#
#p0 = cv2.HoughCircles(old_gray,cv2.HOUGH_GRADIENT,1,20,
#                            param1=50,param2=30,minRadius=0,maxRadius=0)
#p0=p0.transpose(1,0,2)[:,:,:-1]
#p1 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
## Create a mask image for drawing purposes
#mask = np.zeros_like(old_frame)
#cv2.namedWindow( "Display window", cv2.WINDOW_NORMAL );
#for counter,frame_gray in enumerate(m[16000:17000]):
#    print counter
#    frame=cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
#    # calculate optical flow
#    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#
#    # Select good points
#    good_new = p1[st==1]
#    good_old = p0[st==1]
#
#    # draw the tracks
#    for i,(new,old) in enumerate(zip(good_new,good_old)):
#        a,b = new.ravel()
#        c,d = old.ravel()
#        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#    img = cv2.add(frame,mask)
#
#    cv2.imshow("Display window",img)
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break
#
#    # Now update the previous frame and previous points
#    old_gray = frame_gray.copy()
#    p0 = good_new.reshape(-1,1,2)
#
#%%
