import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd

layer = 24
probability = 0.5
carpetas = ['.1_60mm','.01_60mm','.001_60mm','.0001_60mm','.00001_60mm']
brain_number = '1'
carpeta='.001_60mm'
for carpeta in carpetas: 
    
    file_brain = os.path.join('..','Subjects',carpeta,brain_number,'T1.nii')
    brain = nib.load(file_brain).get_fdata()
    brain_cut = brain[:,:,layer]

    file_brain_c = os.path.join('..','Subjects',carpeta,'BiasField_T1.nii')
    brain_c = nib.load(file_brain_c).get_fdata()
    brain_c_cut = brain_c[:,:,layer]

    file_gm = os.path.join('..','Subjects',carpeta,brain_number,'c1T1.nii')
    gm = nib.load(file_gm).get_fdata()
    gm = np.where(gm>probability, 1, 0)
    gm_cut = gm[:,:,layer]

    file = os.path.join('..','Subjects',carpeta,brain_number,'c2T1.nii')
    wm = nib.load(file).get_fdata()
    wm = np.where(wm>probability, 1, 0)
    wm_cut = wm[:,:,layer]

    file = os.path.join('..','Subjects',carpeta,brain_number,'c3T1.nii')
    cfs = nib.load(file).get_fdata()
    cfs = np.where(cfs>probability, 1, 0)
    cfs_cut = cfs[:,:,layer]

    file = os.path.join('..','Subjects',carpeta,brain_number,'c4T1.nii')
    skull = nib.load(file).get_fdata()
    skull = np.where(skull>probability, 1, 0)
    skull_cut = skull[:,:,layer]



    mask = gm_cut + wm_cut + cfs_cut

    resultado = brain_cut*mask

    plt.figure(1)
    plt.imshow(mask,cmap = 'gray')
    plt.title('Brain Mask')
    plt.savefig(f'results/{carpeta}_{brain_number}_brainMasks.png')

    plt.figure(2)
    plt.imshow(brain_cut,cmap = 'gray')
    plt.title('T1')
    plt.savefig(f'results/{carpeta}_{brain_number}T1.png')

    plt.figure(3)
    plt.imshow(resultado,cmap = 'gray')
    plt.title('Skull Striping')
    plt.savefig(f'results/{carpeta}_{brain_number}_SkullStriping.png')

    plt.figure(4)
    plt.imshow(skull_cut,cmap = 'gray')
    plt.title('Skull Mask')
    plt.savefig(f'results/{carpeta}_{brain_number}_skullMasks.png')

    plt.figure(5)
    plt.imshow(gm_cut,cmap = 'gray')
    plt.title('Gray Matter')
    plt.savefig(f'results/{carpeta}_{brain_number}_gm.png')

    plt.figure(6)
    plt.imshow(wm_cut,cmap = 'gray')
    plt.title('White Matter')
    plt.savefig(f'results/{carpeta}_{brain_number}_wm.png')

    plt.figure(7)
    plt.imshow(cfs_cut,cmap = 'gray')
    plt.title('CFS Matter')
    plt.savefig(f'results/{carpeta}_{brain_number}_cfs.png')

    plt.figure(8)
    plt.imshow(brain_c_cut,cmap = 'gray')
    plt.title('Corrected')
    plt.savefig(f'results/{carpeta}_{brain_number}_brain_c.png')
    
    plt.figure(9)
    plt.imshow(brain_c_cut,cmap = 'gray')
    plt.title('Corrected')
    plt.savefig(f'results/{carpeta}_{brain_number}_Diff_brain.png') 