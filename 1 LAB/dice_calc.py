import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd

layer = 24
probability = 0.5
corte = 10
#carpetas = ['.001_40mm','.001_60mm','.001_80mm','.001_100mm','.001_120mm']
#carpetas = ['.1_60mm','.01_60mm','.001_60mm','.0001_60mm','.00001_60mm']
carpetas = ['.1_40mm','.01_40mm','.001_40mm','.0001_40mm']

color_mapping = {
    '.1_40mm': 'red',
    '.01_40mm': 'green',
    '.001_40mm': 'blue',
    '.0001_40mm': 'purple'
}
"""
color_mapping = {
    '.1_60mm': 'red',
    '.01_60mm': 'green',
    '.001_60mm': 'blue',
    '.0001_60mm': 'purple',
    '.00001_60mm': 'brown'
}"""

brain_numbers = ['1','2','3','4','5']

def dice(im1, im2, num_samples=100, confidence_level=0.95):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    dice_coefficient = 2. * intersection.sum() / (im1.sum() + im2.sum())

    return dice_coefficient


gm_data = pd.DataFrame(index=carpetas,columns=brain_numbers)
gm_dice_results, wm_dice_results, csf_dice_results = {}, {}, {}

for carpeta in carpetas:
    tmp, tmp2, tmp3 = [], [], []
    for brain_number in brain_numbers:

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
        
        file = os.path.join('..','Subjects',carpeta,brain_number,'LabelsForTesting.nii')
        cfs_gt = nib.load(file).get_fdata()
        cfs_gt = np.where(cfs_gt==1, 1, 0)
        cfs_gt_cut = cfs_gt[:,:,layer]


        file = os.path.join('..','Subjects',carpeta,brain_number,'LabelsForTesting.nii')
        wm_gt = nib.load(file).get_fdata()
        wm_gt = np.where(wm_gt==3, 1, 0)
        wm_gt_cut = wm_gt[:,:,layer]


        file = os.path.join('..','Subjects',carpeta,brain_number,'LabelsForTesting.nii')
        gm_gt = nib.load(file).get_fdata()
        gm_gt = np.where(gm_gt==2, 1, 0)
        gm_gt_cut = gm_gt[:,:,layer]
        
        dice_CSF = dice(cfs_gt,cfs)
        dice_wm= dice(wm_gt,wm)
        dice_gm= dice(gm_gt,gm)
        
        tmp.append(dice_gm)
        tmp2.append(dice_wm)
        tmp3.append(dice_CSF)
        print(' ********************************************************************')
        print('    ')
        print('Dice Scores of Brain {} with SPM Segementataion with {}'.format(brain_number,carpeta))
        print('    ')
        print('Dice Score from CSF segementation {:.6%}'.format(dice_CSF))
        print('Dice Score from WM segementation {:.6%}'.format(dice_wm))
        print('Dice Score from GM segementation {:.6%}'.format(dice_gm))
        print('    ')
        
    gm_dice_results[carpeta]=tmp
    wm_dice_results[carpeta]=tmp2
    csf_dice_results[carpeta]=tmp3
        
fig, axs = plt.subplots(2, 3)
axs[0, 0].imshow(gm_cut, cmap = 'gray')
axs[0, 0].set_title('Gray Matter with SPM')
axs[0, 1].imshow(wm_cut, cmap = 'gray')
axs[0, 1].set_title('White Matter with SPM')
axs[0, 2].imshow(cfs_cut, cmap = 'gray')
axs[0, 2].set_title('CFS with SPM')

axs[1, 0].imshow(gm_gt_cut, cmap = 'gray')
axs[1, 0].set_title('Gray Matter Gt')
axs[1, 1].imshow(wm_gt_cut, cmap = 'gray')
axs[1, 1].set_title('White Matter Gt')
axs[1, 2].imshow(cfs_gt_cut, cmap = 'gray')
axs[1, 2].set_title('CFS Gt')
fig.savefig(f'results/Brain_{corte}_{layer}.png')
axis_position = plt.axes([0.1, 0, 0.35, 0.03],
                         facecolor = 'White')
slider_position = Slider(axis_position,
                         'Pos', 0, 47, valstep=list(range(48)), valinit=layer)



x_labels = list(gm_dice_results.keys())
y_values = list(gm_dice_results.values())
plt.figure(2)
for i, (label, values) in enumerate(zip(x_labels, y_values)):
    plt.scatter([1,2,3,4,5], values, label=label,color=color_mapping[label], marker='x', s=50, alpha=0.7)

# Add labels and a legend
plt.title('GM Dice Scores')
plt.xticks([1,2,3,4,5])
plt.xlabel('Subjects')
plt.ylabel('Dice Scores')
legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend outside

# Show the plot
plt.tight_layout()  # Ensure everything fits in the plot area
plt.savefig(f'results/GM_{corte}_graph.png')

x2_labels = list(wm_dice_results.keys())
y2_values = list(wm_dice_results.values())
plt.figure(2)
for i, (label, values) in enumerate(zip(x2_labels, y2_values)):
    plt.scatter([1,2,3,4,5], values, label=label,color=color_mapping[label], marker='x', s=50, alpha=0.7)

# Add labels and a legend
plt.title('WM Dice Scores')
plt.xticks([1,2,3,4,5])
plt.xlabel('Subjects')
plt.ylabel('Dice Scores')
legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend outside
# Show the plot
plt.tight_layout()  # Ensure everything fits in the plot area
plt.savefig(f'results/WM_{corte}_graph.png')

x_labels = list(csf_dice_results.keys())
y_values = list(csf_dice_results.values())
plt.figure(2)
for i, (label, values) in enumerate(zip(x_labels, y_values)):
    plt.scatter([1,2,3,4,5], values, label=label,color=color_mapping[label], marker='x', s=50, alpha=0.7)

# Add labels and a legend
plt.title('CFS Dice Scores')
plt.xticks([1,2,3,4,5])
plt.xlabel('Subjects')
plt.ylabel('Dice Scores')
legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend outside

# Show the plot
plt.tight_layout()  # Ensure everything fits in the plot area
plt.savefig(f'results/CFS_{corte}_graph.png')

def updated(val):
    gm_cut = gm[:,:,val]
    axs[0, 0].imshow(gm_cut, cmap = 'gray')
    axs[0, 0].set_title('Gray Matter with SPM')
    gm_gt_cut = gm_gt[:,:,val]
    axs[1, 0].imshow(gm_gt_cut, cmap = 'gray')
    axs[1, 0].set_title('Gray Matter Gt')
        
    wm_cut = wm[:,:,val]
    axs[0, 1].imshow(wm_cut, cmap = 'gray')
    axs[0, 1].set_title('White Matter with SPM')
    wm_gt_cut = wm_gt[:,:,val]
    axs[1, 1].imshow(wm_gt_cut, cmap = 'gray')
    axs[1, 1].set_title('white Matter GT')
    
    cfs_cut = cfs[:,:,val]
    axs[0, 2].imshow(cfs_cut, cmap = 'gray')
    axs[0, 2].set_title('CFS Matter with SPM')
    cfs_gt_cut = cfs_gt[:,:,val]
    axs[1, 2].imshow(cfs_gt_cut, cmap = 'gray')
    axs[1, 2].set_title('CFS Matter GT')
    
    fig.canvas.draw_idle()
    
slider_position.on_changed(updated)
plt.show()



