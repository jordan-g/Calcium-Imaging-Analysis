# -*- coding: utf-8 -*-
"""
Function for implementing parallel scalable segmentation of two photon imaging data

..image::docs/img/cnmf1.png


@author: agiovann
"""
#\package caiman/dource_ectraction/cnmf
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Wed Feb 17 14:58:26 2016

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import numpy as np
import time
import scipy
import os
from ...mmapping import load_memmap
from ...cluster import extract_patch_coordinates


#%%    
def cnmf_patches(args_in):
    """Function that is run for each patches

         Will be called

        Parameters:
        ----------
        file_name: string
            full path to an npy file (2D, pixels x time) containing the movie

        shape: tuple of thre elements
            dimensions of the original movie across y, x, and time

        options:
            dictionary containing all the parameters for the various algorithms

        rf: int
            half-size of the square patch in pixel

        stride: int
            amount of overlap between patches

        gnb: int
            number of global background components

        backend: string
            'ipyparallel' or 'single_thread' or SLURM

        n_processes: int
            nuber of cores to be used (should be less than the number of cores started with ipyparallel)

        memory_fact: double
            unitless number accounting how much memory should be used.
            It represents the fration of patch processed in a single thread.
             You will need to try different values to see which one would work


        Returns:
        -------
        A_tot: matrix containing all the componenents from all the patches

        C_tot: matrix containing the calcium traces corresponding to A_tot

        sn_tot: per pixel noise estimate

        optional_outputs: set of outputs related to the result of CNMF ALGORITHM ON EACH patch

        Raise:
        -----

        Empty Exception
        """

    import logging
    from . import cnmf
    file_name, idx_,shapes,options=args_in

    name_log=os.path.basename(file_name[:-5])+ '_LOG_ ' + str(idx_[0])+'_'+str(idx_[-1])
    logger = logging.getLogger(name_log)
    hdlr = logging.FileHandler('./'+name_log)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)

    p=options['temporal_params']['p']

    logger.info('START')

    logger.info('Read file')
    Yr, dims, timesteps = load_memmap(file_name)

    # slicing array (takes the min and max index in n-dimensional space and cuts the box they define)
    # for 2d a rectangle/square, for 3d a rectangular cuboid/cube, etc.
    upper_left_corner = min(idx_)
    lower_right_corner = max(idx_)
    indices = np.unravel_index([upper_left_corner, lower_right_corner], dims, order='F')  # indices as tuples
    slices = [slice(min_dim, max_dim + 1) for min_dim, max_dim in indices]
    slices.insert(0, slice(timesteps))  # insert slice for timesteps, equivalent to :

    images = np.reshape(Yr.T, [timesteps] + list(dims), order='F')
    images = images[slices]

    if (np.sum(np.abs(np.diff(images.reshape(timesteps, -1).T)))) > 0.1:

        cnm = cnmf.CNMF(n_processes = 1, k = options['init_params']['K'], gSig = options['init_params']['gSig'],
                merge_thresh = options['merging']['thr'], p = p, dview = None,  Ain = None,  Cin = None,
                f_in = None, do_merge = True,
                ssub = options['init_params']['ssub'], tsub = options['init_params']['tsub'],
                p_ssub = options['patch_params']['ssub'], p_tsub = options['patch_params']['tsub'],
                method_init = options['init_params']['method'], alpha_snmf = options['init_params']['alpha_snmf'],
                rf=None,stride=None, memory_fact=1, gnb = options['init_params']['nb'],
                only_init_patch = options['patch_params']['only_init'],
                method_deconvolution =  options['temporal_params']['method'],
                n_pixels_per_process = options['preprocess_params']['n_pixels_per_process'],
                block_size = options['temporal_params']['block_size'],
                check_nan = options['preprocess_params']['check_nan'],
                skip_refinement = options['patch_params']['skip_refinement'],
                options_local_NMF = options['init_params']['options_local_NMF'],
                normalize_init = options['init_params']['normalize_init'],
                remove_very_bad_comps = options['patch_params']['remove_very_bad_comps'])

        cnm = cnm.fit(images)
        return idx_,shapes,scipy.sparse.coo_matrix(cnm.A),\
               cnm.b,cnm.C,cnm.f,cnm.S,cnm.bl,cnm.c1,\
               cnm.neurons_sn,cnm.g,cnm.sn,cnm.options,cnm.YrA.T
    else:
        return None


#%%
def run_CNMF_patches(file_name, shape, options, rf=16, stride = 4, gnb = 1, dview=None, memory_fact=1):
    """Function that runs CNMF in patches

     Either in parallel or sequentially, and return the result for each.
     It requires that ipyparallel is running

     Will basically initialize everything in order to compute on patches then call a function in parallel that will
     recreate the cnmf object and fit the values.
     It will then recreate the full frame by listing all the fitted values together

    Parameters:
    ----------        
    file_name: string
        full path to an npy file (2D, pixels x time) containing the movie        

    shape: tuple of thre elements
        dimensions of the original movie across y, x, and time 

    options:
        dictionary containing all the parameters for the various algorithms

    rf: int 
        half-size of the square patch in pixel

    stride: int
        amount of overlap between patches

    gnb: int
        number of global background components

    backend: string
        'ipyparallel' or 'single_thread' or SLURM

    n_processes: int
        nuber of cores to be used (should be less than the number of cores started with ipyparallel)

    memory_fact: double
        unitless number accounting how much memory should be used.
        It represents the fration of patch processed in a single thread.
         You will need to try different values to see which one would work


    Returns:
    -------
    A_tot: matrix containing all the components from all the patches

    C_tot: matrix containing the calcium traces corresponding to A_tot

    sn_tot: per pixel noise estimate

    optional_outputs: set of outputs related to the result of CNMF ALGORITHM ON EACH patch

    Raise:
    -----

    Empty Exception
    """
    dims=shape[:-1]
    d = np.prod(dims)
    T = shape[-1]
    
    if np.isscalar(rf):
        rfs=[rf]*len(dims)
    else:
        rfs = rf
        
    if np.isscalar(stride):    
        strides = [stride]*len(dims)
    else:
        strides = stride
        
    options['preprocess_params']['n_pixels_per_process']=np.int(old_div(np.prod(rfs),memory_fact))
    options['spatial_params']['n_pixels_per_process']=np.int(old_div(np.prod(rfs),memory_fact))
    options['temporal_params']['n_pixels_per_process']=np.int(old_div(np.prod(rfs),memory_fact))
    nb = options['spatial_params']['nb']

    idx_flat,idx_2d=extract_patch_coordinates(dims, rfs, strides)
    args_in=[]
    for id_f,id_2d in zip(idx_flat,idx_2d):        
        print(id_2d)
        args_in.append((file_name, id_f,id_2d, options))

    st=time.time()
    if dview is not None:
        try:
            file_res = dview.map_sync(cnmf_patches, args_in)
            dview.results.clear()
        except:
            print('Something went wrong')  
            raise
        finally:
            print('You may think that it went well but reality is harsh')

    else:
        file_res = list(map(cnmf_patches, args_in))                         

    print((time.time()-st))
    # count components
    count=0
    count_bgr = 0
    patch_id=0
    num_patches=len(file_res)
    for fff in file_res:
        if fff is not None:
            idx_,shapes,A,b,C,f,S,bl,c1,neurons_sn,g,sn,_,YrA=fff
            for _ in range(np.shape(b)[-1]):
                count_bgr += 1

            for ii in range(np.shape(A)[-1]):            
                
                new_comp=scipy.sparse.csc_matrix(old_div(A.tocsc()[:,ii],np.sqrt(np.sum(
                    np.array(A.tocsc()[:,ii].todense())**2))))
                if new_comp.sum()>0:
                    count+=1

            patch_id+=1

    #INITIALIZING
    C_tot=np.zeros((count,T))
    YrA_tot=np.zeros((count,T))
    F_tot=np.zeros((nb*num_patches,T))
    mask=np.zeros(d)
    sn_tot=np.zeros((d))

    f_tot ,bl_tot ,c1_tot ,neurons_sn_tot ,g_tot ,idx_tot ,id_patch_tot ,shapes_tot = [],[],[],[],[],[],[],[]
    patch_id, empty, count_bgr, count = 0,0,0,0
    idx_tot_B, idx_tot_A, a_tot, b_tot = [],[],[],[]
    idx_ptr_B, idx_ptr_A = [0],[0]

    # instead of filling in the matrices, construct lists with their non-zero entries and coordinates
    print('Transforming patches into full matrix')
    for fff in file_res:
        if fff is not None:

            idx_,shapes,A,b,C,f,S,bl,c1,neurons_sn,g,sn,_,YrA = fff

            sn_tot[idx_]=sn
            f_tot.append(f)
            bl_tot.append(bl)
            c1_tot.append(c1)
            neurons_sn_tot.append(neurons_sn)
            g_tot.append(g)
            idx_tot.append(idx_)
            shapes_tot.append(shapes)
            mask[idx_] += 1

            for ii in range(np.shape(b)[-1]):
                b_tot.append(b[:,ii])
                idx_tot_B.append(idx_)
                idx_ptr_B.append(len(idx_))
                F_tot[patch_id,:]=f[ii,:]
                count_bgr += 1

            for ii in range(np.shape(A)[-1]):            
                new_comp=old_div(A.tocsc()[:,ii],np.sqrt(np.sum(np.array(A.tocsc()[:,ii].todense())**2)))
                if new_comp.sum()>0:
                    a_tot.append(new_comp.toarray().flatten())
                    idx_tot_A.append(idx_)
                    idx_ptr_A.append(len(idx_))
                    C_tot[count,:]=C[ii,:]                      
                    YrA_tot[count,:]=YrA[ii,:]
                    id_patch_tot.append(patch_id)
                    count+=1

            patch_id+=1  
        else:
            empty+=1
            
    print('Skipped %d Empty Patch',empty)
    idx_tot_B = np.concatenate(idx_tot_B)
    b_tot = np.concatenate(b_tot)     
    idx_ptr_B = np.cumsum(np.array(idx_ptr_B))
    B_tot = scipy.sparse.csc_matrix((b_tot, idx_tot_B, idx_ptr_B), shape=(d, count_bgr))     
  
    idx_tot_A = np.concatenate(idx_tot_A)
    a_tot = np.concatenate(a_tot)     
    idx_ptr_A = np.cumsum(np.array(idx_ptr_A))
    A_tot = scipy.sparse.csc_matrix((a_tot, idx_tot_A, idx_ptr_A), shape=(d, count))         

    C_tot=C_tot[:count,:]
    YrA_tot=YrA_tot[:count,:]  

    optional_outputs=dict()
    optional_outputs['b_tot']=b_tot
    optional_outputs['f_tot']=f_tot
    optional_outputs['bl_tot']=bl_tot
    optional_outputs['c1_tot']=c1_tot
    optional_outputs['neurons_sn_tot']=neurons_sn_tot
    optional_outputs['g_tot']=g_tot
    optional_outputs['idx_tot']=idx_tot
    optional_outputs['shapes_tot']=shapes_tot
    optional_outputs['id_patch_tot']= id_patch_tot
    optional_outputs['B'] = B_tot
    optional_outputs['F'] = F_tot
    optional_outputs['mask'] = mask

    print("Generating background")
    Im = scipy.sparse.csr_matrix((old_div(1.,mask),(np.arange(d),np.arange(d))))
    Bm = Im.dot(B_tot)
    A_tot = Im.dot(A_tot)

    f = np.r_[np.atleast_2d(np.mean(F_tot,axis=0)),np.random.rand(gnb-1,T)]

    for _ in range(100):
        b = np.fmax(Bm.dot(F_tot.dot(f.T)).dot(np.linalg.inv(f.dot(f.T))),0)
        f = np.fmax(np.linalg.inv(b.T.dot(b)).dot((Bm.T.dot(b)).T.dot(F_tot)),0)

    return A_tot,C_tot,YrA_tot,b,f,sn_tot, optional_outputs


