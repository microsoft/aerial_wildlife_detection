'''
    RetinaNet trainer for PyTorch.

    2019 Benjamin Kellenberger
'''

import io
from tqdm import tqdm
from celery import current_task
import torch
from torch.utils.data import DataLoader

from ..genericPyTorchModel import GenericPyTorchModel
from .. import parse_transforms

from ..functional._retinanet import DEFAULT_OPTIONS, collation, encoder, loss
from ..functional._retinanet.model import RetinaNet as Model
from ..functional.datasets.bboxDataset import BoundingBoxesDataset
from util.helpers import get_class_executable, check_args



class RetinaNet(GenericPyTorchModel):

    model_class = Model

    def __init__(self, config, dbConnector, fileServer, options):
        super(RetinaNet, self).__init__(config, dbConnector, fileServer, options, DEFAULT_OPTIONS)

        # set defaults if not explicitly overridden
        if self.model_class is None:
            self.model_class = Model
        if self.criterion_class is None:
            self.criterion_class = loss.FocalLoss
        if self.dataset_class is None:
            self.dataset_class = BoundingBoxesDataset


    def train(self, stateDict, data):
        '''
            Initializes a model based on the given stateDict and a data loader from the
            provided data and trains the model, taking into account the parameters speci-
            fied in the 'options' given to the class.
            Returns a serializable state dict of the resulting model.
        '''

        # initialize model
        model, labelclassMap = self.initializeModel(stateDict, data)

        # setup transform, data loader, dataset, optimizer, criterion
        inputSize = tuple(self.options['general']['image_size'])
        transform = parse_transforms(self.options['train']['transform'])
        
        dataset = self.dataset_class(data=data,
                                    fileServer=self.fileServer,
                                    labelclassMap=labelclassMap,
                                    targetFormat='xyxy',
                                    transform=transform,
                                    ignoreUnsure=self.options['train']['ignore_unsure'])
        dataEncoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   #TODO: implement into options
        collator = collation.Collator((inputSize[1], inputSize[0],), dataEncoder)
        dataLoader = DataLoader(
            dataset=dataset,
            collate_fn=collator.collate_fn,
            **self.options['train']['dataLoader']['kwargs']
        )

        # optimizer
        optimizer = self.optim_class(params=model.parameters(), **self.options['train']['optim']['kwargs'])

        # loss criterion
        criterion = self.criterion_class(**self.options['train']['criterion']['kwargs'])

        # train model
        device = self.get_device()
        torch.manual_seed(self.options['general']['seed'])
        if 'cuda' in device:
            torch.cuda.manual_seed(self.options['general']['seed'])
        model.to(device)
        imgCount = 0
        for (img, bboxes_target, labels_target, fVec, _) in tqdm(dataLoader):
            img, bboxes_target, labels_target = img.to(device), \
                                                bboxes_target.to(device), \
                                                labels_target.to(device)

            optimizer.zero_grad()
            bboxes_pred, labels_pred = model(img)
            loss_value = criterion(bboxes_pred, bboxes_target, labels_pred, labels_target)
            loss_value.backward()
            optimizer.step()
            
            # update worker state
            imgCount += img.size(0)
            current_task.update_state(state='PROGRESS', meta={'done': imgCount, 'total': len(dataLoader.dataset), 'message': 'training'})

        # all done; return state dict as bytes
        return self.exportModelState(model)

    
    def inference(self, stateDict, data):
        '''
            TODO
        '''

        # initialize model
        if stateDict is None:
            raise Exception('No trained model state found, but required for inference.')

        # read state dict from bytes
        model, labelclassMap = self.initializeModel(stateDict, data)

        # initialize data loader, dataset, transforms
        inputSize = tuple(self.options['general']['image_size'])
        transform = parse_transforms(self.options['inference']['transform'])
        
        dataset = self.dataset_class(data=data,
                                    fileServer=self.fileServer,
                                    labelclassMap=labelclassMap,
                                    transform=transform)
        dataEncoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   #TODO: ditto
        collator = collation.Collator((inputSize[1], inputSize[0],), dataEncoder)
        dataLoader = DataLoader(
            dataset=dataset,
            collate_fn=collator.collate_fn,
            **self.options['inference']['dataLoader']['kwargs']
        )

        # perform inference
        response = {}
        device = self.get_device()
        model.to(device)
        imgCount = 0
        for (img, _, _, fVec, imgID) in tqdm(dataLoader):

            # TODO: implement feature vectors
            # if img is not None:
            #     dataItem = img.to(device)
            #     isFeatureVector = False
            # else:
            #     dataItem = fVec.to(device)
            #     isFeatureVector = True
            dataItem = img.to(device)

            with torch.no_grad():
                bboxes_pred_batch, labels_pred_batch = model(dataItem, False)   #TODO: isFeatureVector
                bboxes_pred_batch, labels_pred_batch, confs_pred_batch = dataEncoder.decode(bboxes_pred_batch.squeeze(0).cpu(),
                                    labels_pred_batch.squeeze(0).cpu(),
                                    inputSize,
                                    cls_thresh=0.1, nms_thresh=0.1,
                                    numPred_max=self.options['inference']['num_predictions_max'],
                                    return_conf=True)       #TODO: ditto

                for i in range(len(imgID)):
                    bboxes_pred = bboxes_pred_batch[i]
                    labels_pred = labels_pred_batch[i]
                    confs_pred = confs_pred_batch[i]
                    if bboxes_pred.dim() == 2:
                        bboxes_pred = bboxes_pred.unsqueeze(0)
                        labels_pred = labels_pred.unsqueeze(0)
                        confs_pred = confs_pred.unsqueeze(0)

                    # convert bounding boxes to YOLO format
                    predictions = []
                    bboxes_pred_img = bboxes_pred[0,...]
                    labels_pred_img = labels_pred[0,...]
                    confs_pred_img = confs_pred[0,...]
                    if len(bboxes_pred_img):
                        bboxes_pred_img[:,2] -= bboxes_pred_img[:,0]
                        bboxes_pred_img[:,3] -= bboxes_pred_img[:,1]
                        bboxes_pred_img[:,0] += bboxes_pred_img[:,2]/2
                        bboxes_pred_img[:,1] += bboxes_pred_img[:,3]/2
                        bboxes_pred_img[:,0] /= inputSize[0]
                        bboxes_pred_img[:,1] /= inputSize[1]
                        bboxes_pred_img[:,2] /= inputSize[0]
                        bboxes_pred_img[:,3] /= inputSize[1]

                        # limit to image bounds
                        bboxes_pred_img = torch.clamp(bboxes_pred_img, 0, 1)


                        # append to dict
                        for b in range(bboxes_pred_img.size(0)):
                            bbox = bboxes_pred_img[b,:]
                            label = labels_pred_img[b]
                            logits = confs_pred_img[b,:]
                            predictions.append({
                                'x': bbox[0].item(),
                                'y': bbox[1].item(),
                                'width': bbox[2].item(),
                                'height': bbox[3].item(),
                                'label': dataset.labelclassMap_inv[label.item()],
                                'logits': logits.numpy().tolist(),        #TODO: for AL criterion?
                                'confidence': torch.max(logits).item()
                            })
                    
                    response[imgID[i]] = {
                        'predictions': predictions,
                        #TODO: exception if fVec is not torch tensor: 'fVec': io.BytesIO(fVec.numpy().astype(np.float32)).getvalue()
                    }

            # update worker state
            imgCount += len(imgID)
            current_task.update_state(state='PROGRESS', meta={'done': imgCount, 'total': len(dataLoader.dataset), 'message': 'predicting'})

        model.cpu()
        if 'cuda' in device:
            torch.cuda.empty_cache()

        return response