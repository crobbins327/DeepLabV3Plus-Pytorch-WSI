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

mask_dir = r'E:\Applikate\Kidney-DeepLearning\Kidney CNN File 1.6.2021\cell-masks'
save_dir = r'E:\Applikate\Kidney-DeepLearning\Kidney CNN File 1.6.2021\cell-masks\masks\128x128_stride64'
os.makedirs(save_dir, exist_ok=True)
masks = [p for p in os.listdir(mask_dir) if p.endswith('.png')]
stride=64
patchsize=128
f = open('E:\Applikate\Kidney-DeepLearning\Kidney CNN File 1.6.2021\cell-masks\classes.json')
classes = json.load(f)
#Removing these classes for each patch...
remove_classes = ['Use Patch', 'Myofibroblast', 'Smooth muscle', 'Partial Nuclei', 
                  'Glomerular Nuclei', 'Interstitial Nuclei', 'Vascular Nuclei']
keep_classes = list(set(classes.keys())-set(remove_classes))
#Reorder the class list, make the list for the new values
keep_classes = [k for k in classes if k in keep_classes]
val_classes = np.arange(0,len(keep_classes),1)
val_classes = np.uint8(255*val_classes/len(val_classes)).tolist()
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
      
      #get coordinates and save patch, these are the coords of the downsampled image
      #Might be off by a couple pixels for when converting to original coords...
      patch_name = '{}_coord{},{}_ds{}.png'.format(wsi_name,x,y,ds)
      Image.fromarray(patch, 'L').save(os.path.join(save_dir, patch_name))
      #Another precaution to make sure that the new patch loaded does not have values replaced
      del raw_patch, patch

with open(os.path.join(save_dir,"classes.json"), "w") as outfile:
    json.dump(keep_classes, outfile, indent=1)