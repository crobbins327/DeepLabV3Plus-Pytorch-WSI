from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from collections import namedtuple

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from datasets import VOCSegmentation, Cityscapes
from datasets import WSIMaskDataset
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--mask_dir", type=str, help="path to segmentation masks for WSI datasets")
    parser.add_argument("--dataset", type=str, default='KID-MP-10cell',
                        choices=['KID-MP-3cell',
                                 'KID-MP-6cell',
                                 'KID-MP-8cell',
                                 'KID-MP-10cell',
                                 'KID-MP-13cell'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument('--results_dir', type=str, default='./results', 
                        help='where results are saved')
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--res", type=int, default=512)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: cross_entropy)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use tensorboard for visualization")
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if 'KID-MP' in opts.dataset:
        # parser = argparse.ArgumentParser()
        # args = parser.parse_args()
        # args.n_gpu = 4
        # args.batch = 16
        wsi_dir = '/home/cjr66/project/KID-DeepLearning/KID-Images-pyramid'
        coord_dir = None
        # mask_dir = '/home/cjr66/project/KID-DeepLearning/Labeled_patches/MP_256x256_stride64'
        process_list = '/home/cjr66/project/KID-DeepLearning/proc_info/MP_only-KID_process_list.csv'
        # rescale_mpp = True
        desired_mpp = 0.2
        wsi_exten = ['.tif','.svs']
        mask_exten = '.png'
        
        #Could load in separate folds for cross validation instead of generating splits...
        #Generate train, val, test splits
        from sklearn.model_selection import train_test_split
        import pandas as pd
        
        mask_files = sorted([x for x in os.listdir(opts.mask_dir) if x.endswith(mask_exten)])
        #0.7, 0.15, 0.15 splits
        y = 0.15
        seed=27
        m_train, m_test = train_test_split(mask_files, test_size=y, random_state=seed)
        m_train, m_val = train_test_split(m_train, test_size=y/(1-y), random_state=seed)
        #Save splits to file?
        os.makedirs(opts.results_dir, exist_ok=True)
        split_dict = {'train_split': m_train,
                      'val_split': m_val,
                      'test_split': m_test
                      }
        split_df = pd.DataFrame.from_dict(split_dict, orient='index').T
        split_df.to_csv(os.path.join(opts.results_dir,'{}train-{}val-{}test_split{}.csv'.format(1-2*y,y,y,seed)), index=False)
        
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
            classes = [
                KIDCellClass('Background',      0,  0,  0, 'Background', 0, False, False, (0, 0, 0)),
                KIDCellClass('Lymphocytes',    19,  1,  1, 'Nuclei', 1, True, False, (0, 128, 0)),
                KIDCellClass('Neutrophils',    39,  2,  1, 'Nuclei', 1, True, False, (0, 255, 0)),
                KIDCellClass('Macrophage',     58,  3, 255, 'Nuclei', 1, True, True, (255, 153, 102)),
                KIDCellClass('PCT Nuclei',     78,  4,  2, 'Nuclei', 1, True, False, (255, 0, 255)),
                KIDCellClass('DCT Nuclei',     98,  5,  3, 'Nuclei', 1, True, False, (0, 0, 128)),
                KIDCellClass('Endothelial',   117,  6,  4, 'Nuclei', 1, True, False, (0, 128, 128)),
                KIDCellClass('Fibroblast',    137,  7,  5, 'Nuclei', 1, True, False, (235, 206, 155)),
                KIDCellClass('Mesangial',     156,  8,  6, 'Nuclei', 1, True, False, (255, 255, 0)),    
                KIDCellClass('Parietal cells',176,  9,  7, 'Nuclei', 1, True, False, (58, 208, 67)),    
                KIDCellClass('Podocytes',     196, 10, 8, 'Nuclei', 1, True, False, (0, 255, 255)),  
                KIDCellClass('Mitosis',       215, 11, 255, 'Nuclei', 1, True, True, (179, 26, 26)),   
                KIDCellClass('Tubule Nuclei', 235, 12, 9, 'Nuclei', 1, True, False, (130, 91, 37)),    
                ]
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
            
        train_ids, index = np.unique(np.array([c.train_id for c in classes]), return_index=True)
        opts.num_classes = len(train_ids[train_ids!=255])
        print('Number of classes: {}'.format(opts.num_classes))
        train_dst = WSIMaskDataset(opts, wsi_dir, coord_dir, opts.mask_dir, classes=classes, 
                                 process_list = process_list,
                                 wsi_exten=wsi_exten, mask_exten=mask_exten,
                                 rescale_mpp=True, desired_mpp=desired_mpp, 
                                 is_label=True,
                                 phase='train',
                                 mask_split_list=m_train,
                                 aug=True,
                                 resolution=opts.res,
                                 one_hot=False,
                                 make_all_pipelines=True)
        
        val_dst = WSIMaskDataset(opts, wsi_dir, coord_dir, opts.mask_dir, classes=classes, 
                                 process_list = process_list,
                                 wsi_exten=wsi_exten, mask_exten=mask_exten, 
                                 rescale_mpp=True, desired_mpp=desired_mpp, 
                                 is_label=True,
                                 phase='val',
                                 mask_split_list=m_val,
                                 aug=False, 
                                 resolution=opts.res,
                                 one_hot=False,
                                 make_all_pipelines=False)
    
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, denorm = None, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists(opts.results_dir):
            os.makedirs(opts.results_dir, exist_ok=True)
        # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])
        if denorm is None:
            denorm = utils.Denormalize(mean=None, std=None, skip=True)
        else:
            assert(isinstance(denorm, utils.Denormalize))
            
        img_id = 0

    with torch.no_grad():
        # for i, (images, labels) in tqdm(enumerate(loader)):
        for i, seg_data in tqdm(enumerate(loader)):
            
            images, labels = seg_data['image'], seg_data['mask']
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            if labels.dim() == 4:
                labels = labels.squeeze(1)

            outputs = model(images)
            # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            preds = outputs.detach().cpu()
            preds = torch.argmax(preds, dim=1).to(dtype=torch.int).numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save(os.path.join(opts.results_dir,'{}_image.png'.format(img_id)))
                    Image.fromarray(target).save(os.path.join(opts.results_dir,'{}_target.png'.format(img_id)))
                    Image.fromarray(pred).save(os.path.join(opts.results_dir,'{}_pred.png'.format(img_id)))
                    
                    #Replace with color_mask_overlay function somehow...
                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig(os.path.join(opts.results_dir,'{}_overlay.png'.format(img_id)), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    
    #Make results directories
    os.makedirs(opts.results_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.results_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(opts.results_dir, 'checkpoints'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opts.results_dir, 'logs'))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, 
        num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))
    
    #debugging scheduler
    # parser = argparse.ArgumentParser()
    # opts = parser.parse_args()
    # opts.num_classes = 10
    # opts.output_stride = 16
    # opts.model = 'deeplabv3plus_resnet101'
    # opts.separable_conv = True
    
    # Set up model (all models are 'constructed at network.modeling)
    print('Setting up:', opts.model)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    # Change out optimizer to ADAM?
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    # utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            #Edit total_itrs in scheduler state that is being loaded
            checkpoint["scheduler_state"]["max_iters"] = opts.total_itrs
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
    denorm = utils.Denormalize(mean=None, std=None, skip=True)  # denormalization for ori images
    
    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, denorm=denorm, 
            ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        # for (images, labels) in train_loader:
        for seg_data in train_loader:
            cur_itrs += 1

            images, labels = seg_data['image'], seg_data['mask']
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            #Prepare labels which are not one hot encoded
            if labels.dim() == 4:
                labels = labels.squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            #Expand dimensions for categorical cross entropy? 
            
            #leave it alone for sparse cross entropy (pytorch default)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            writer.add_scalar('train/loss',np_loss, global_step=cur_itrs)
            writer.add_scalar('train/backbone_lr',scheduler.get_lr()[0], global_step=cur_itrs)
            writer.add_scalar('train/classifier_lr',scheduler.get_lr()[1], global_step=cur_itrs)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                writer.add_scalar('train/interval_loss',interval_loss, global_step=cur_itrs)
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt(os.path.join(opts.results_dir, 'checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride)))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    denorm = denorm, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(os.path.join(opts.results_dir, 'checkpoints/best_%s_%s_os%d.pth' % 
                                           (opts.model, opts.dataset, opts.output_stride)))
                    
                    
                writer.add_scalar('val/overall_acc',val_score['Overall Acc'], global_step=cur_itrs)
                writer.add_scalar('val/mean_acc',val_score['Mean Acc'], global_step=cur_itrs)
                writer.add_scalar('val/freqw_acc',val_score['FreqW Acc'], global_step=cur_itrs)
                writer.add_scalar('val/mean_IoU',val_score['Mean IoU'], global_step=cur_itrs)
                print('Class IoU:')
                print(val_score['Class IoU'])
                keys_values = val_score['Class IoU'].items()
                class_ious = {str(key): value for key, value in keys_values}
                writer.add_scalars('val/class_IoU',class_ious, global_step=cur_itrs)
                
                #Limit saving images to save space in TF event file....
                if (cur_itrs) % (5 * opts.val_interval) == 0:
                    for k, (img, target, lbl) in enumerate(ret_samples):
                        #CxHxW -> HxWxC for PIL
                        img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)
                        target = train_dst.decode_target(target).astype(np.uint8)
                        color_lbl = train_dst.decode_target(lbl).astype(np.uint8)
                        color_overlay = train_dst.color_mask_overlay(img, lbl, a=0.5).astype(np.uint8)
                        concat_img = np.concatenate((img, target, color_lbl, color_overlay), axis=1)  # concat along width
                    
                        writer.add_image('sample_{}'.format(k), concat_img, global_step=cur_itrs, dataformats='HWC')

                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
