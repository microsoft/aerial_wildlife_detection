'''
    RetinaNet trainer for PyTorch.

    2019 Benjamin Kellenberger
'''

import io
from tqdm import tqdm
from celery import current_task
import tensorflow as tf
import numpy as np
#from torch.utils.data import DataLoader

from ..genericTFModel import GenericTFModel
from .. import parse_transforms

from ..functional._yolo_3 import DEFAULT_OPTIONS, collation, encoder, loss
from ..functional._yolo_3.model import yolo_model as Model
from ..functional.datasets.bboxDataset import BoundingBoxesDataset
from util.helpers import get_class_executable, check_args



class yolo(GenericTFModel):

    model_class = Model

    def __init__(self, config, dbConnector, fileServer, options):
        super(yolo, self).__init__(config, dbConnector, fileServer, options, DEFAULT_OPTIONS)

        # set defaults if not explicitly overridden
        if self.model_class is None:
            self.model_class = Model
        if self.criterion_class is None:
            self.criterion_class = loss.YoloLoss
        if self.dataset_class is None:
            self.dataset_class = BoundingBoxesDataset


    def train(self, stateDict, data):
        '''
            Initializes a model based on the given stateDict and a data loader from the
            provided data and trains the model, taking into account the parameters speci-
            fied in the 'options' given to the class.
            Returns a serializable state dict of the resulting model.
        '''

        inputSize = tuple((self.options['train']['width'], self.options['train']['height']))
        # initialize model
        model, labelclassMap = self.initializeModel(stateDict, data, inputSize[0], inputSize[1])

#
#        # setup transform, data loader, dataset, optimizer, criterion
        transform = parse_transforms(self.options['train']['transform'])
        dataEncoder = encoder.DataEncoder(numClasses=len(labelclassMap.keys()))  
        
        dataset = self.dataset_class(data=data,
                                    fileServer=self.fileServer,
                                    labelclassMap=labelclassMap,
                                    targetFormat='xywh',
                                    transform=transform,
                                    ignoreUnsure=self.options['train']['ignore_unsure'],
                                    batch_size=self.options['train']['batch_size'],
                                    encoder = dataEncoder.encode)

        # optimizer
        print(self.options['train']['optim']['kwargs'])
        optimizer = self.optim_class(**self.options['train']['optim']['kwargs'])

        # loss criterion
        loss = self.criterion_class(inputSize[0], inputSize[1], **self.options['train']['criterion']['kwargs'])
        model.yolo_nn.compile(loss=loss.yolo_loss, optimizer=optimizer)

        epochs = (self.options['train']['epochs'] if 'epochs' in self.options['train'] else 1)
        model.yolo_nn.fit_generator(generator = dataset, epochs = epochs) 
#
#        # train model
#        device = self.get_device()
#        torch.manual_seed(self.options['general']['seed'])
#        if 'cuda' in device:
#            torch.cuda.manual_seed(self.options['general']['seed'])
#        model.to(device)
#        imgCount = 0
#        for (img, bboxes_target, labels_target, fVec, _) in tqdm(dataLoader):
#            img, bboxes_target, labels_target = img.to(device), \
#                                                bboxes_target.to(device), \
#                                                labels_target.to(device)
#
#            optimizer.zero_grad()
#            bboxes_pred, labels_pred = model(img)
#            loss_value = criterion(bboxes_pred, bboxes_target, labels_pred, labels_target)
#            loss_value.backward()
#            optimizer.step()
#            
#            # update worker state
#            imgCount += img.size(0)
#            current_task.update_state(state='PROGRESS', meta={'done': imgCount, 'total': len(dataLoader.dataset), 'message': 'training'})
#
#        # all done; return state dict as bytes
        return self.exportModelState(model)
#
    
    def inference(self, stateDict, data):
        '''
        
        '''

        if stateDict is None:
            raise Exception('No trained model state found, but required for inference.')

        # initialize model
        inputSize = tuple((self.options['inference']['width'], self.options['inference']['height']))
        model, labelclassMap = self.initializeModel(stateDict, data, inputSize[0], inputSize[1])

        # initialize data loader, dataset, transforms
        transform = parse_transforms(self.options['inference']['transform'])
        
        dataset = self.dataset_class(data=data,
                                    fileServer=self.fileServer,
                                    labelclassMap=labelclassMap,
                                    transform=transform,
                                    batch_size=self.options['inference']['batch_size'],
                                    shuffle=self.options['inference']['shuffle'])

        dataEncoder = encoder.DataEncoder(numClasses=len(labelclassMap.keys()))  

        cls_thresh = self.options['inference']['cls_thresh']
        nms_thresh = self.options['inference']['nms_thresh']

        # perform inference
        response = {}
        imgCount = 0
        for (img, _, _, imgID) in tqdm(dataset):

            output = model.yolo_nn.predict(img)
            bboxes_pred_batch, labels_pred_batch, confs_pred_batch = dataEncoder.decode(output, cls_thresh=cls_thresh, nms_thresh=nms_thresh, return_conf=True) 


            for i in range(len(imgID)):
                bboxes_pred = bboxes_pred_batch[i]
                labels_pred = labels_pred_batch[i]
                confs_pred = confs_pred_batch[i]

                # convert bounding boxes to YOLO format
                predictions = []
                if len(bboxes_pred):
                    bboxes_pred[:,2] -= bboxes_pred[:,0]
                    bboxes_pred[:,3] -= bboxes_pred[:,1]
                    bboxes_pred[:,0] += bboxes_pred[:,2]/2
                    bboxes_pred[:,1] += bboxes_pred[:,3]/2
                    bboxes_pred[:,0] /= inputSize[0]
                    bboxes_pred[:,1] /= inputSize[1]
                    bboxes_pred[:,2] /= inputSize[0]
                    bboxes_pred[:,3] /= inputSize[1]

                    # limit to image bounds
                    bboxes_pred[bboxes_pred<0.0]=0.0
                    bboxes_pred[bboxes_pred>1.0]=1.0



                    # append to dict
                    for b in range(bboxes_pred.shape[0]):
                        bbox = bboxes_pred[b,:]
                        label = labels_pred[b]
                        logits = confs_pred[b,:]
                        predictions.append({
                            'x': bbox[0],
                            'y': bbox[1],
                            'width': bbox[2],
                            'height': bbox[3],
                            'label': dataset.labelclassMap_inv[int(label)],
                            'logits': logits.tolist(),        #TODO: for AL criterion?
                            'confidence': np.max(logits)
                        })
                
                response[imgID[i]] = {
                    'predictions': predictions
            
                }

            # update worker state
            imgCount += len(imgID)
            current_task.update_state(state='PROGRESS', meta={'done': imgCount, 'total': len(dataset), 'message': 'predicting'})


        return response
