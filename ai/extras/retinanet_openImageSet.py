'''
    RetinaNet model with the inference function replaced:
    Here, we do inference on full images, split into patches,
    instead of going through data already in the database.

    2019 Benjamin Kellenberger
'''

import os
import io
import glob
import re
import random
import numpy as np
from tqdm import tqdm
from celery import current_task
import torch
from torchvision import transforms as tr
from PIL import Image
from ..models.pytorch.detection.retinanet import RetinaNet
from ai.models.pytorch.functional._retinanet.model import RetinaNet as Model
from ai.models.pytorch.functional._retinanet import encoder
from ai.models.pytorch.functional._retinanet.utils import box_nms
import ai.models.pytorch.functional._util.bboxTransforms as bboxTr
from ._functional import tensorSharding, windowCropping


class RetinaNet_ois(RetinaNet):

    def __init__(self, config, dbConnector, fileServer, options):
        super(RetinaNet_ois, self).__init__(config, dbConnector, fileServer, options)

        # parameters
        self.batchSize = self.options['contrib']['batch_size']
        self.maxNumUnlabeled = self.options['contrib']['inference_max_num_unlabeled']
        self.patchSize = tuple(self.options['general']['image_size'])
        self.stride = self.options['contrib']['stride']
        self.encoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   #TODO: implement 

        self.windowCropper = windowCropping.WindowCropper(
            patchSize=self.patchSize, exportEmptyPatches=False,
            cropMode='windowCropping',
            searchStride=(10,10,),
            minBBoxArea=64, minBBoxAreaFrac=64  #TODO
        )

        # extra: parse base folder for images to look out for
        self.__parse_base_folder()

    
    def __parse_base_folder(self):
        self.baseFolder_unlabeled = self.options['contrib']['baseFolder_unlabeled']
        self.loadRaw = self.options['contrib']['load_raw_images']

        self.all_images = []

        # retrieve all images
        generators = [
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.JPG'), recursive=True),
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.jpg'), recursive=True),
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.JPEG'), recursive=True),
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.jpeg'), recursive=True),
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.PNG'), recursive=True),
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.png'), recursive=True)
        ]
        if self.loadRaw:
            generators.append(glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.NEF'), recursive=True))
            generators.append(glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.nef'), recursive=True))
            generators.append(glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.CR2'), recursive=True))
            generators.append(glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.cr2'), recursive=True))

        for gen in generators:
            try:
                while True:
                    imgPath = next(gen)
                    self.all_images.append(imgPath)
            except:
                # end of generator
                pass

    

    def _inference_image(self, model, transform, filename):
        '''
            Loads the image with given filename from disk, splits it up into
            regular patches, performs inference and then re-splits the image
            into patches that fit the predicted boxes tightly
            ('WindowCropping' strategy), if there are any boxes.
            Commits the resulting patch names to the database and returns the
            identified bounding boxes under the patch names as a dict.
        '''

        # load image
        img = Image.open(io.BytesIO(self.fileServer.getFile(filename)))

        # transform
        tensor = transform(img).to(self._get_device())

        # evaluate in a grid fashion
        gridX, gridY = tensorSharding.createSplitLocations_auto(img.size, [self.patchSize[1], self.patchSize[0]], stride=self.stride, tight=True)
        tensors = tensorSharding.splitTensor(tensor, [self.patchSize[1], self.patchSize[0]], gridY, gridX)
        gridX, gridY = gridX.view(-1).float(), gridY.view(-1).float()

        bboxes = torch.empty(size=(0,4,), dtype=torch.float32)
        labels = torch.empty(size=(0,), dtype=torch.long)
        confs = torch.empty(size=(0, model.numClasses,), dtype=torch.float32)
        scores = torch.empty(size=(0,), dtype=torch.float32)

        numPatches = tensors.size(0)
        numBatches = int(np.ceil(numPatches / float(self.batchSize)))
        for t in range(numBatches):
            startIdx = t*self.batchSize
            endIdx = min((t+1)*self.batchSize, numPatches)
            
            batch = tensors[startIdx:endIdx,:,:,:]
            
            if len(batch.size())==3:
                batch = batch.unsqueeze(0)

            with torch.no_grad():
                bboxes_pred_img, labels_pred_img = model(batch)
            
            bboxes_pred_img, labels_pred_img, confs_pred_img = self.encoder.decode(bboxes_pred_img.squeeze(0).cpu(),
                                                labels_pred_img.squeeze(0).cpu(),
                                                self.patchSize,
                                                cls_thresh=0.1, nms_thresh=0,    #TODO
                                                return_conf=True)

            # incorporate patch offsets and append to list of predictions
            if len(bboxes_pred_img):
                bboxes_pred_img[:,0] += gridX[startIdx:endIdx]
                bboxes_pred_img[:,1] += gridY[startIdx:endIdx]
                bboxes_pred_img[:,2] += gridX[startIdx:endIdx]
                bboxes_pred_img[:,3] += gridY[startIdx:endIdx]

                scores_pred_img, _ = torch.max(confs_pred_img,1)

                bboxes = torch.cat((bboxes, bboxes_pred_img), dim=0)
                labels = torch.cat((labels, labels_pred_img), dim=0)
                confs = torch.cat((confs, confs_pred_img), dim=0)
                scores = torch.cat((scores, scores_pred_img), dim=0)

        # do NMS on entire set
        keep = box_nms(bboxes, scores, threshold=0.1)   #TODO
        bboxes = bboxes[keep,:]
        labels = labels[keep]
        confs = confs[keep,:]
        scores = scores[keep]

        # re-split into patches (WindowCropping)
        patchData = self.windowCropper.splitImageIntoPatches(img, bboxes, labels, confs)

        # iterate over patches
        result = {}
        for key in patchData.keys():
            # patch name
            patchName = re.sub('\..*$', '', filename) + '_' + key + os.path.splitext(filename)[1]

            # save patch
            bytea = io.BytesIO()
            patchData[key]['patch'].save(bytea)
            self.fileServer.putfile(bytea.getvalue(), patchName)    #TODO: verify

            # append metadata
            result[patchName] = {
                'predictions': patchData[key]['predictions']
            }

        # return metadata
        return result


    def inference(self, stateDict, data):
        '''
            Augmented implementation of RetinaNet's regular inference function.
            In addition to (or instead of, depending on the settings) performing
            inference on images already existing in the database, the model runs
            over large images specified in the folder ('all_images') and adds the
            predicted patches to the database.
            TODO: Requires the FileServer to be running on the same instance.
        '''

        # prepare return metadata
        response = {}

        # initialize model
        if stateDict is None:
            raise Exception('No trained model state found, but required for inference.')

        # read state dict from bytes
        stateDict = io.BytesIO(stateDict)
        stateDict = torch.load(stateDict, map_location=lambda storage, loc: storage)
        model = Model.loadFromStateDict(stateDict)

        # get all image filenames from DB
        current_task.update_state(state='PREPARING', meta={'message':'identifying images'})
        sql = 'SELECT filename FROM {schema}.image;'.format(schema=self.config.getProperty('Database', 'schema'))
        filenames = self.dbConnector.execute(sql, None, 'all')

        #TODO
        from celery.contrib import rdb
        rdb.set_trace()

        # get valid filename substring (pattern: path/base_x_y_w_h.JPG)
        fileSnippets_db = set([re.sub('_[0-9]+_[0-9]+_[0-9]+_[0-9]+\..*$', '', f) for f in filenames])

        # the same for images on disk
        fileSnippets_disk = set([re.sub('\..*$', '', f.replace(self.baseFolder_unlabeled, '')) for f in self.all_images])

        # identify images that have not yet been added to DB
        unlabeled = fileSnippets_disk.intersection(fileSnippets_db)

        if not len(unlabeled):
            return response

        # choose n unlabeled
        unlabeled = random.sample(unlabeled, min(self.maxNumUnlabeled, len(unlabeled)))

        # prepare transforms
        transform = bboxTr.Compose([
            bboxTr.DefaultTransform(tr.ToTensor()),
            bboxTr.DefaultTransform(tr.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]))
        ])  #TODO: write functional.pytorch util to compose transformations

        # do inference on the unlabeled images
        for u in tqdm(range(len(unlabeled))):
            meta = self._inference_image(model, transform, unlabeled[u])
            for key in meta.keys():
                response[key] = meta[key]

            # update worker state
            current_task.update_state(state='PROGRESS', meta={'done': u+1, 'total': len(unlabeled), 'message': 'predicting'})
    
        model.cpu()
        if 'cuda' in self._get_device():
            torch.cuda.empty_cache()

        return response