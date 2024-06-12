import numpy as np
from collections import namedtuple
import torch

class KIDCellDataset:
    
    def __init__(self, opts):

        self.classif_labels = None

        if  '-3cell' in opts.dataset:
            KIDCellClass = namedtuple('KIDCellClass', ['name', 'mask_value', 'id', 'train_id', 'category', 'category_id',
                                                            'has_instances', 'ignore_in_eval', 'color'])
            #ignore index is 255
            classes = [
                KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
                KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
                KIDCellClass('Neutrophils',    39,  2,  1, 'Nuclei', 1, True, False, (0, 255, 0)),
                KIDCellClass('Macrophage',     58,  3, 255, 'Nuclei', 1, True, True, (255, 153, 102)),
                KIDCellClass('PCT Nuclei',     78,  4,  2, 'Nuclei', 1, True, False, (255, 0, 255)),
                KIDCellClass('DCT Nuclei',     98,  5,  2, 'Nuclei', 1, True, False, (0, 0, 128)),
                KIDCellClass('Endothelial',   117,  6,  2, 'Nuclei', 1, True, False, (0, 128, 128)),
                KIDCellClass('Fibroblast',    137,  7,  2, 'Nuclei', 1, True, False, (235, 206, 155)),
                KIDCellClass('Mesangial',     156,  8,  2, 'Nuclei', 1, True, False, (255, 255, 0)),    
                KIDCellClass('Parietal cells',176,  9,  2, 'Nuclei', 1, True, False, (58, 208, 67)),    
                KIDCellClass('Podocytes',     196, 10, 2, 'Nuclei', 1, True, False, (0, 255, 255)),  
                KIDCellClass('Mitosis',       215, 11, 255, 'Nuclei', 1, True, True, (179, 26, 26)),   
                KIDCellClass('Tubule Nuclei', 235, 12, 2, 'Nuclei', 1, True, False, (130, 91, 37)),    
                ]
        elif  '-6cell' in opts.dataset:
            KIDCellClass = namedtuple('KIDCellClass', ['name', 'mask_value', 'id', 'train_id', 'category', 'category_id',
                                                            'has_instances', 'ignore_in_eval', 'color'])
            #ignore index is 255
            classes = [
                KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
                KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
                KIDCellClass('Neutrophils',    39,  2,  1, 'Nuclei', 1, True, False, (0, 255, 0)),
                KIDCellClass('Macrophage',     58,  3, 255, 'Nuclei', 1, True, True, (255, 153, 102)),
                KIDCellClass('PCT Nuclei',     78,  4,  2, 'Nuclei', 1, True, False, (255, 0, 255)),
                KIDCellClass('DCT Nuclei',     98,  5,  2, 'Nuclei', 1, True, False, (0, 0, 128)),
                KIDCellClass('Endothelial',   117,  6,  3, 'Nuclei', 1, True, False, (0, 128, 128)),
                KIDCellClass('Fibroblast',    137,  7,  4, 'Nuclei', 1, True, False, (235, 206, 155)),
                KIDCellClass('Mesangial',     156,  8,  5, 'Nuclei', 1, True, False, (255, 255, 0)),    
                KIDCellClass('Parietal cells',176,  9,  5, 'Nuclei', 1, True, False, (58, 208, 67)),    
                KIDCellClass('Podocytes',     196, 10, 5, 'Nuclei', 1, True, False, (0, 255, 255)),  
                KIDCellClass('Mitosis',       215, 11, 255, 'Nuclei', 1, True, True, (179, 26, 26)),   
                KIDCellClass('Tubule Nuclei', 235, 12, 2, 'Nuclei', 1, True, False, (130, 91, 37)),    
                ]
        elif  '-8cell' in opts.dataset:
            KIDCellClass = namedtuple('KIDCellClass', ['name', 'mask_value', 'id', 'train_id', 'category', 'category_id',
                                                            'has_instances', 'ignore_in_eval', 'color'])
            #ignore index is 255
            classes = [
                KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
                KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
                KIDCellClass('Neutrophils',    39,  2,  1, 'Nuclei', 1, True, False, (0, 255, 0)),
                KIDCellClass('Macrophage',     58,  3, 255, 'Nuclei', 1, True, True, (255, 153, 102)),
                KIDCellClass('PCT Nuclei',     78,  4,  2, 'Nuclei', 1, True, False, (255, 0, 255)),
                KIDCellClass('DCT Nuclei',     98,  5,  2, 'Nuclei', 1, True, False, (0, 0, 128)),
                KIDCellClass('Endothelial',   117,  6,  3, 'Nuclei', 1, True, False, (0, 128, 128)),
                KIDCellClass('Fibroblast',    137,  7,  4, 'Nuclei', 1, True, False, (235, 206, 155)),
                KIDCellClass('Mesangial',     156,  8,  5, 'Nuclei', 1, True, False, (255, 255, 0)),    
                KIDCellClass('Parietal cells',176,  9,  6, 'Nuclei', 1, True, False, (58, 208, 67)),    
                KIDCellClass('Podocytes',     196, 10, 7, 'Nuclei', 1, True, False, (0, 255, 255)),  
                KIDCellClass('Mitosis',       215, 11, 255, 'Nuclei', 1, True, True, (179, 26, 26)),   
                KIDCellClass('Tubule Nuclei', 235, 12, 2, 'Nuclei', 1, True, False, (130, 91, 37)),    
                ]
        elif  '-10cell' in opts.dataset:
            KIDCellClass = namedtuple('KIDCellClass', ['name', 'mask_value', 'id', 'train_id', 'category', 'category_id',
                                                            'has_instances', 'ignore_in_eval', 'color'])
            #ignore index is 255
            # classes = [
            #     KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
            #     KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
            #     KIDCellClass('Neutrophils',    39,  2,  1, 'Nuclei', 1, True, False, (0, 255, 0)),
            #     KIDCellClass('Macrophage',     58,  3, 255, 'Nuclei', 1, True, True, (255, 153, 102)),
            #     KIDCellClass('PCT Nuclei',     78,  4,  2, 'Nuclei', 1, True, False, (255, 0, 255)),
            #     KIDCellClass('DCT Nuclei',     98,  5,  3, 'Nuclei', 1, True, False, (0, 0, 128)),
            #     KIDCellClass('Endothelial',   117,  6,  4, 'Nuclei', 1, True, False, (0, 128, 128)),
            #     KIDCellClass('Fibroblast',    137,  7,  5, 'Nuclei', 1, True, False, (235, 206, 155)),
            #     KIDCellClass('Mesangial',     156,  8,  6, 'Nuclei', 1, True, False, (255, 255, 0)),    
            #     KIDCellClass('Parietal cells',176,  9,  7, 'Nuclei', 1, True, False, (58, 208, 67)),    
            #     KIDCellClass('Podocytes',     196, 10, 8, 'Nuclei', 1, True, False, (0, 255, 255)),  
            #     KIDCellClass('Mitosis',       215, 11, 255, 'Nuclei', 1, True, True, (179, 26, 26)),   
            #     KIDCellClass('Tubule Nuclei', 235, 12, 9, 'Nuclei', 1, True, False, (130, 91, 37)),    
            #     ]
            
            # combine tubule nuclei and DCT nuclei
            # classes = [
            #     KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
            #     KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
            #     KIDCellClass('Neutrophils',    39,  2,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
            #     KIDCellClass('Macrophage',     58,  3, 255, 'Nuclei', 1, True, True, (0, 128, 0)),
            #     KIDCellClass('PCT Nuclei',     78,  4,  2, 'Nuclei', 1, True, False, (0, 0, 128)),
            #     KIDCellClass('DCT Nuclei',     98,  5,  3, 'Nuclei', 1, True, False, (0, 0, 128)),
            #     KIDCellClass('Endothelial',   117,  6,  4, 'Nuclei', 1, True, False, (0, 128, 128)),
            #     KIDCellClass('Fibroblast',    137,  7,  5, 'Nuclei', 1, True, False, (235, 206, 155)),
            #     KIDCellClass('Mesangial',     156,  8,  6, 'Nuclei', 1, True, False, (255, 255, 0)),    
            #     KIDCellClass('Parietal cells',176,  9,  7, 'Nuclei', 1, True, False, (58, 208, 67)),    
            #     KIDCellClass('Podocytes',     196, 10, 8, 'Nuclei', 1, True, False, (0, 255, 255)),  
            #     KIDCellClass('Mitosis',       215, 11, 255, 'Nuclei', 1, True, True, (0, 0, 128)),   
            #     KIDCellClass('Tubule Nuclei', 235, 12, 9, 'Nuclei', 1, True, False, (0, 0, 128)),    
            #     ]

            # combine similar classes by output color
            classes = [
                KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
                KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
                KIDCellClass('Neutrophils',    39,  2,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
                KIDCellClass('Macrophage',     58,  3, 255, 'Nuclei', 1, True, True, (0, 128, 0)),
                KIDCellClass('PCT Nuclei',     78,  4,  2, 'Nuclei', 1, True, False, (0, 0, 128)),
                KIDCellClass('DCT Nuclei',     98,  5,  3, 'Nuclei', 1, True, False, (0, 0, 128)),
                KIDCellClass('Endothelial',   117,  6,  4, 'Nuclei', 1, True, False, (128, 0, 0)),
                KIDCellClass('Fibroblast',    137,  7,  5, 'Nuclei', 1, True, False, (235, 206, 155)),
                KIDCellClass('Mesangial',     156,  8,  6, 'Nuclei', 1, True, False, (0, 255, 255)),    
                KIDCellClass('Parietal cells',176,  9,  7, 'Nuclei', 1, True, False, (58, 208, 67)),    
                KIDCellClass('Podocytes',     196, 10, 8, 'Nuclei', 1, True, False, (255, 255, 0)),  
                KIDCellClass('Mitosis',       215, 11, 255, 'Nuclei', 1, True, True, (0, 0, 128)),   
                KIDCellClass('Tubule Nuclei', 235, 12, 9, 'Nuclei', 1, True, False, (0, 0, 128)),    
                ]
            
            # hack for reclassifying output classes
            classif_labels = {
                1: ['Lymphocytes', (0, 128, 0)],
                2: ['Tubule Nuclei', (0, 0, 128)],
                3: ['Tubule Nuclei', (0, 0, 128)],
                4: ['Endothelial', (128, 0, 0)],
                5: ['Fibroblast', (235, 206, 155)],
                6: ['Mesangial', (0, 255, 255)],
                7: ['Parietal', (58, 208, 67)],
                8: ['Podocytes', (255, 255, 0)],
                9: ['Tubule Nuclei', (0, 0, 128)],
            }

            self.classif_labels = classif_labels
            
            # leukocyte vs other
            # classes = [
            #     KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
            #     KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
            #     KIDCellClass('Neutrophils',    39,  2,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
            #     KIDCellClass('Macrophage',     58,  3, 255, 'Nuclei', 1, True, True, (0, 128, 0)),
            #     KIDCellClass('PCT Nuclei',     78,  4,  2, 'Nuclei', 1, True, False, (235, 206, 155)),
            #     KIDCellClass('DCT Nuclei',     98,  5,  3, 'Nuclei', 1, True, False, (235, 206, 155)),
            #     KIDCellClass('Endothelial',   117,  6,  4, 'Nuclei', 1, True, False, (235, 206, 155)),
            #     KIDCellClass('Fibroblast',    137,  7,  5, 'Nuclei', 1, True, False, (235, 206, 155)),
            #     KIDCellClass('Mesangial',     156,  8,  6, 'Nuclei', 1, True, False, (235, 206, 155)),    
            #     KIDCellClass('Parietal cells',176,  9,  7, 'Nuclei', 1, True, False, (235, 206, 155)),    
            #     KIDCellClass('Podocytes',     196, 10, 8, 'Nuclei', 1, True, False, (235, 206, 155)),  
            #     KIDCellClass('Mitosis',       215, 11, 255, 'Nuclei', 1, True, True, (235, 206, 155)),   
            #     KIDCellClass('Tubule Nuclei', 235, 12, 9, 'Nuclei', 1, True, False, (235, 206, 155)),    
            #     ]
            
        elif  '-13cell' in opts.dataset:
            KIDCellClass = namedtuple('KIDCellClass', ['name', 'mask_value', 'id', 'train_id', 'category', 'category_id',
                                                            'has_instances', 'ignore_in_eval', 'color'])
            #ignore index is 255
            classes = [
                KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
                KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
                KIDCellClass('Neutrophils',    39,  2,  2, 'Nuclei', 1, True, False, (0, 255, 0)),
                KIDCellClass('Macrophage',     58,  3,  3, 'Nuclei', 1, True, True, (255, 153, 102)),
                KIDCellClass('PCT Nuclei',     78,  4,  4, 'Nuclei', 1, True, False, (255, 0, 255)),
                KIDCellClass('DCT Nuclei',     98,  5,  5, 'Nuclei', 1, True, False, (0, 0, 128)),
                KIDCellClass('Endothelial',   117,  6,  6, 'Nuclei', 1, True, False, (0, 128, 128)),
                KIDCellClass('Fibroblast',    137,  7,  7, 'Nuclei', 1, True, False, (235, 206, 155)),
                KIDCellClass('Mesangial',     156,  8,  8, 'Nuclei', 1, True, False, (255, 255, 0)),    
                KIDCellClass('Parietal cells',176,  9,  9, 'Nuclei', 1, True, False, (58, 208, 67)),    
                KIDCellClass('Podocytes',     196, 10, 10, 'Nuclei', 1, True, False, (0, 255, 255)),  
                KIDCellClass('Mitosis',       215, 11, 11, 'Nuclei', 1, True, True, (179, 26, 26)),   
                KIDCellClass('Tubule Nuclei', 235, 12, 12, 'Nuclei', 1, True, False, (130, 91, 37)),    
                ]
        else:
            raise ValueError('{} not an implemented dataset!'.format(opts.dataset))
        
        self.classes = classes
        if self.classif_labels is None:
            # construct simple classification labels based on train id in classes object
            classif_labels = {}
            for i, c in enumerate(classes):
                classif_labels[c.train_id] = [c.name, c.color]
            self.classif_labels = classif_labels
        
        #For preparing the masks to seg ID images
        train_ids, index = np.unique(np.array([c.train_id for c in classes]), return_index=True)
        colors = [c.color for c in classes]
        self.train_id_to_color = [colors[index[i]] for i,t in enumerate(train_ids) if (t != -1 and t != 255)]
        #Color for ignore IDs
        self.train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = np.array(self.train_id_to_color)
        
        self.id_to_train_id = np.array(([c.train_id for c in classes]))
        self.val_to_id_dict = dict([(c.mask_value, c.id) for c in classes])
        self.val_to_train_id_dict = dict([(c.mask_value, c.train_id) for c in classes])
        
        self.num_classes = len(train_ids[train_ids!=255])


    def decode_target(self, mask_seg):
        """decode semantic mask to RGB image"""
        #Set ignore ids to ignore color at end of array
        mask_seg[mask_seg == 255] = len(self.train_id_to_color)-1
        
        return self.train_id_to_color[mask_seg]
    
    def mask_val_to_seg(self, mask_np):
        val_keys = list(self.val_to_id_dict.keys())
        mask_seg = np.zeros((1, mask_np.shape[0], mask_np.shape[1]))
        for i in range(len(val_keys)):
            mask_seg[0][mask_np==val_keys[i]] = self.val_to_id_dict[val_keys[i]]
        #Then convert IDs to train IDs
        mask_seg = mask_seg.astype(int)
        mask_seg = self.id_to_train_id[mask_seg]
        return mask_seg
    
    def color_mask_overlay(self, img, mask_seg, a=0.5):
        if type(img) == torch.Tensor:
            masked = img.detach().clone()
            for key in range(self.num_classes):
                if key != 0:
                    masked[mask_seg==key] = torch.tensor(self.train_id_to_color[key], dtype=torch.float)
            color_overlay = masked*a + img*(1-a)
            return color_overlay.numpy().astype(np.uint8)
        elif type(img) == np.ndarray:
            masked = np.copy(img)
            for key in range(self.num_classes):
                if key != 0:
                    masked[mask_seg==key] = self.train_id_to_color[key]
            color_overlay = masked*a + img*(1-a)
            return np.array(color_overlay).astype(np.uint8)
        else:
            raise ValueError('img of type {} not handled in color_mask_overlay()...'.format(type(img)))