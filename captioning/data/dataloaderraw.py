from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import torch
import skimage
import skimage.io
import scipy.misc
import sys

from torchvision import transforms as trn
import pickle
from copy import deepcopy
from PIL import Image

preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from ..utils.resnet_utils import myResnet
from ..utils import resnet
#from ..super_resolution.model.common import resolve_single

class DataLoaderRaw():
    
    def __init__(self, opt):
        self.opt = opt
        self.coco_json = opt.get('coco_json', '')
        self.folder_path = opt.get('folder_path', '')

        self.batch_size = opt.get('batch_size', 1)
        self.seq_per_img = 1
        # Pass Super Resolution model
        self.sr_model = opt.get('sr_model', None)

        # Load resnet
        self.cnn_model = opt.get('cnn_model', 'resnet101')
        self.my_resnet = getattr(resnet, self.cnn_model)()
        self.my_resnet.load_state_dict(torch.load('./data/imagenet_weights/'+self.cnn_model+'.pth'))
        self.my_resnet = myResnet(self.my_resnet)
        self.my_resnet.cuda()
        self.my_resnet.eval()



        # load the json file which contains additional information about the dataset
        print('DataLoaderRaw loading images from folder: ', self.folder_path)

        self.files = []
        self.ids = []

        print(len(self.coco_json))

        # read in all the filenames from the folder
        print('listing all images in directory ' + self.folder_path)
        def isImage(f):
            supportedExt = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.ppm','.PPM']
            for ext in supportedExt:
                start_idx = f.rfind(ext)
                if start_idx >= 0 and start_idx + len(ext) == len(f):
                    return True
            return False

        # n = 1
        # for root, dirs, files in os.walk(self.folder_path, topdown=False):
        #     for file in files:
        #         fullpath = os.path.join(self.folder_path, file)
        #         if isImage(fullpath):
        #             self.files.append(fullpath)
        #             self.ids.append(str(n)) # just order them sequentially
        #             n = n + 1
        # self.N = len(self.files)

        frame2data_path = os.path.join(self.folder_path, 'frame2data.pkl')
        self.frame2data = pickle.load(open(frame2data_path, 'rb'))
        self.N = int(max(self.frame2data))
        print('DataLoaderRaw found ', self.N, ' frames')

        self.iterator = 0

        # Nasty
        self.dataset = self  # to fix the bug in eval

    def get_batch(self, split, batch_size=None):
        # batch_size = min(batch_size or self.batch_size, self.N)
        while str(self.iterator) not in self.frame2data:
            self.iterator += 1
        frame_num = str(self.iterator)
        # {path: frame_path, rois: [roi_1, ..., roi_n], ...}
        frame_data = self.frame2data[frame_num]
        batch_size = len(frame_data['rois'])
        # pick an index of the datapoint to load next
        fc_batch = np.ndarray((batch_size, 2048), dtype = 'float32')
        att_batch = np.ndarray((batch_size, 14, 14, 2048), dtype = 'float32')
        max_index = self.N
        wrapped = False
        infos = []
        # print('Batch Size: {}'.format(batch_size))
        # Ensure frame2data index always passes. Don't fail on missing frames.
        print('Batch Size for Frame {} is: {}'.format(frame_num, batch_size))
        # img = Image.open(frame_data['path'])
        img = skimage.io.imread(frame_data['path'])
        for i in range(batch_size):
            # ri = self.iterator
            # ri_next = ri + 1
            # if ri_next >= max_index:
                # Reached End of File, Break
                # break
                # ri_next = 0
                # wrapped = True
                # wrap back around
            # self.iterator = ri_next

            # img = skimage.io.imread(self.files[ri])

            roi_bbox = frame_data['rois'][i].astype(np.int)
            print('BBOX | Shape: {} | Max Val: {}'.format(roi_bbox.shape, np.max(roi_bbox)))
            print('ROI BBox: {}'.format(roi_bbox))
            # cropped_img = img[roi_bbox[1]:roi_bbox[3]+1, roi_bbox[0]:roi_bbox[2]+1, :]
            cropped_img = img[roi_bbox[0]:roi_bbox[2]+1, roi_bbox[1]:roi_bbox[3]+1, :]
            # cropped_img = deepcopy(img).crop(
            #     (roi_bbox[0], roi_bbox[1], roi_bbox[2]+1, roi_bbox[3]+1)
            # )
            # Apply Super-Resolution to Image
            if self.sr_model is not None:
                cropped_img = resolve_single(self.sr_model, cropped_img)

            if len(cropped_img.shape) == 2:
                cropped_img = cropped_img[:,:,np.newaxis]
                cropped_img = np.concatenate((cropped_img, cropped_img, cropped_img), axis=2)

            cropped_img = cropped_img[:,:,:3].astype('float32')/255.0
            cropped_img = torch.from_numpy(cropped_img.transpose([2,0,1])).cuda()
            cropped_img = preprocess(cropped_img)
            with torch.no_grad():
                tmp_fc, tmp_att = self.my_resnet(cropped_img)

            fc_batch[i] = tmp_fc.data.cpu().float().numpy()
            att_batch[i] = tmp_att.data.cpu().float().numpy()

            info_struct = {}
            info_struct['id'] = i
            info_struct['file_path'] = os.path.join(os.path.dirname(frame_data['path']),
                                                    '{}_{}.jpg'.format(frame_num, i))
            infos.append(info_struct)

        data = {}
        data['fc_feats'] = fc_batch
        data['att_feats'] = att_batch.reshape(batch_size, -1, 2048)
        data['att_masks'] = None
        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': self.N, 'wrapped': wrapped}
        data['infos'] = infos

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data

    def reset_iterator(self, split):
        self.iterator = 0

    def get_vocab_size(self):
        return len(self.ix_to_word)

    def get_vocab(self):
        return self.ix_to_word
        
