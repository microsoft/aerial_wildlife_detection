'''
    Generic AIWorker model implementation that supports PyTorch models for classification.

    2019 Benjamin Kellenberger
'''

import io
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from celery import current_task

from .. import GenericPyTorchModel, parse_transforms, get_device
from . import DEFAULT_OPTIONS

from util.helpers import get_class_executable, check_args



class ClassificationModel(GenericPyTorchModel):

    def __init__(self, config, dbConnector, fileServer, options):
        super(ClassificationModel, self).__init__(config, dbConnector, fileServer, options, DEFAULT_OPTIONS)


    def train(self, stateDict, data):
        '''
            Initializes a model based on the given stateDict and a data loader from the
            provided data and trains the model, taking into account the parameters speci-
            fied in the 'options' given to the class.
            Returns a serializable state dict of the resulting model.
        '''

        model, labelclassMap = self.initializeModel(stateDict, data)

        # # initialize model
        # if stateDict is not None:
        #     stateDict = torch.load(io.BytesIO(stateDict), map_location=lambda storage, loc: storage)
        #     model = self.model_class.loadFromStateDict(stateDict)
            
        #     # mapping labelclass (UUID) to index in model (number)
        #     labelclassMap = stateDict['labelclassMap']
        # else:
        #     # create new label class map
        #     labelclassMap = {}
        #     for index, lcID in enumerate(data['labelClasses']):
        #         labelclassMap[lcID] = index
        #     self.options['model']['labelclassMap'] = labelclassMap

        #     # initialize a fresh model
        #     model = self.model_class.loadFromStateDict(self.options['model']['kwargs'])


        # setup transform, data loader, dataset, optimizer, criterion
        transform = parse_transforms(self.options['train']['transform'])

        dataset = self.dataset_class(data, self.fileServer, labelclassMap,
                                transform, self.options['train']['ignore_unsure'],
                                **self.options['dataset']['kwargs']
                                )

        dataLoader = DataLoader(dataset,
                                **self.options['train']['dataLoader']['kwargs']
                                )

        optimizer_class = get_class_executable(self.options['train']['optim']['class'])
        optimizer = optimizer_class(params=model.parameters(), **self.options['train']['optim']['kwargs'])

        criterion_class = get_class_executable(self.options['train']['criterion']['class'])
        criterion = criterion_class(**self.options['train']['criterion']['kwargs'])

        # train model
        device = get_device(self.options)
        torch.manual_seed(self.options['general']['seed'])
        if 'cuda' in device:
            torch.cuda.manual_seed(self.options['general']['seed'])

        model.to(device)
        imgCount = 0
        for (img, bboxes_target, labels_target, fVec, _) in tqdm(dataLoader):
            img, bboxes_target, labels_target = img.to(device), bboxes_target.to(device), labels_target.to(device)

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
        # if 'cuda' in device:
        #     torch.cuda.empty_cache()
        # model.cpu()

        # bio = io.BytesIO()
        # torch.save(model.getStateDict(), bio)

        # return bio.getvalue()


    
    def inference(self, stateDict, data):
        '''
            Initializes a model based on the given stateDict and a data loader from the
            provided data and performs inference on the images, taking into account the
            parameters specified in the 'options' given to the class.
            Returns a dict of the images provided, augmented with model predictions.
        '''

        # initialize model
        if stateDict is None:
            raise Exception('No trained model state found, but required for inference.')

        # read state dict from bytes
        model, labelclassMap = self.initializeModel(stateDict, data)

        # stateDict = io.BytesIO(stateDict)
        # stateDict = torch.load(stateDict, map_location=lambda storage, loc: storage)
        # model = self.model_class.loadFromStateDict(stateDict)

        # # mapping labelClass (UUID) to index in model (number)
        # labelclassMap = stateDict['labelclassMap']

        # setup transform, data loader, dataset, optimizer, criterion
        transform = parse_transforms(self.options['inference']['transform'])

        dataset = self.dataset_class(data, self.fileServer, labelclassMap,
                                transform, False,
                                **self.options['dataset']['kwargs']
                                )

        dataLoader = DataLoader(dataset,
                                **self.options['inference']['dataLoader']['kwargs']
                                )

        # perform inference
        response = {}
        model.to(self.device)
        imgCount = 0
        for (img, _, _, fVec, imgID) in tqdm(dataLoader):

            dataItem = img.to(self.device)
            with torch.no_grad():
                pred_batch = model(dataItem)
            
            # append to dict
            for i in range(len(imgID)):
                logits = pred_batch[i,:]
                label = torch.argmax(logits).item()
                
                response[imgID[i]] = {
                    'predictions': [
                        {
                            'label': dataset.labelclassMap_inv[label],
                            'logits': logits.numpy().tolist(),        #TODO: for AL criterion?
                            'confidence': torch.max(logits).item()
                        }
                    ]
                }
        
            # update worker state
            imgCount += len(imgID)
            current_task.update_state(state='PROGRESS', meta={'done': imgCount, 'total': len(dataLoader.dataset), 'message': 'predicting'})

        model.cpu()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()

        return response