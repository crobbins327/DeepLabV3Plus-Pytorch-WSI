# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 23:06:19 2022

@author: jackr
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
Image.MAX_IMAGE_PIXELS = None
from mat_utils import nuclei_dict_from_mask
import scipy.io as sio

mask_dir = 'E:/Applikate/Kidney-DeepLearning/Kidney CNN File 1.6.2021/cell-masks'
save_dir = 'E:/Applikate/Kidney-DeepLearning/Kidney CNN File 1.6.2021/cell-masks/mats/512x512_stride128'
os.makedirs(save_dir, exist_ok=True)
masks = [p for p in os.listdir(mask_dir) if p.endswith('.png')]
stride=128
patchsize=512
f = open('E:/Applikate/Kidney-DeepLearning/Kidney CNN File 1.6.2021/cell-masks/classes.json')
classes = json.load(f)
#Removing these classes for each patch...
remove_classes = ['Use Patch']
keep_classes = list(set(classes.keys())-set(remove_classes))
#Reorder the class list, make the list for the new values
keep_classes = [k for k in classes if k in keep_classes]
val_classes = np.arange(1,len(keep_classes)+1,1)
val_classes = np.uint8(255*val_classes/(len(val_classes)+1)).tolist()
#Combine keep_classes and val_classes into dict
keep_classes = dict(zip(keep_classes,val_classes))

for file in masks:
  # wsi_name = file.split('_cells-cell-mask-1ds.png')[0]
  if '_cells-cell-mask' in file:
    wsi_name = file.split('_cells-cell-mask')[0]
  elif '-cell-mask' in file:
    wsi_name = file.split('-cell-mask')[0]
  ds = file.split('cell-mask-')[1].split('ds.png')[0]
  tissue_mask = np.array(Image.open(os.path.join(mask_dir,file))) 
  h,w = tissue_mask.shape
  for x in range(0, w, stride):
    for y in range(0, h, stride):
      #get patch
      raw_patch = tissue_mask[y:y+patchsize,x:x+patchsize]
      #determine if patch contains > 80% Use Patch and keep_classes
      #count nonzero is fine in this case
      pFill = np.count_nonzero(raw_patch)/raw_patch.size
      if pFill < 0.9:
        # print('Skipping patch...')
        continue
      print('Saving patch {},{}...'.format(x,y))
      #Need to create copy so that replacing does not permantly change patch values....
      patch = raw_patch.copy()
      #Change patch values using keep_classes and val_classes
      for j in classes:
        if j in remove_classes:
          # print('Removing class {}'.format(j))
          patch[patch==classes[j]] = 0
        if j in keep_classes:
          # print('Changing class {}, {} to {}'.format(j, classes[j], keep_classes[j]))
          patch[patch==classes[j]] = keep_classes[j]
      
      meta = {
        'wsiname': wsi_name,
        'patch_size': patchsize,
        'stride': stride,
        'x': x,
        'y': y,
        'ds': ds
      }
      nuclei_dict = nuclei_dict_from_mask(patch, metadata=meta, border_val=255)
      # limit to 65535 instances
      # if len(nuclei_dict['id']) > 65535:
      if nuclei_dict['inst_map'].max() > 2**16-1:
        # print('Warning: too many instances in patch, truncating to 65535')
        raise ValueError('Too many instances in patch')
      nuclei_dict['inst_map'] = nuclei_dict['inst_map'].astype(np.uint16)
      # limit to 255 classes
      nuclei_dict['class_map'] = nuclei_dict['class_map'].astype(np.uint8)
      
      #get coordinates and save patch, these are the coords of the downsampled image
      patch_name = '{}_coord{},{}_ds{}.mat'.format(wsi_name,x,y,ds)
      sio.savemat(os.path.join(save_dir, patch_name), nuclei_dict, do_compression=True)
      # test_load = sio.loadmat(os.path.join(save_dir, patch_name))
      #Another precaution to make sure that the new patch loaded does not have values replaced
      del raw_patch, patch

with open(os.path.join(save_dir,"classes.json"), "w") as outfile:
    json.dump(keep_classes, outfile, indent=1)