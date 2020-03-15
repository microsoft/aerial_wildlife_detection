'''
    Generic AIWorker model implementation that supports PyTorch models for classification.

    2019-20 Benjamin Kellenberger
'''

import io
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..genericPyTorchModel import GenericPyTorchModel
from .. import parse_transforms
from ._default_options import DEFAULT_OPTIONS
from ..functional.classification.collation import collate

from util.helpers import get_class_executable, check_args



class ClassificationModel(GenericPyTorchModel):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(ClassificationModel, self).__init__(project, config, dbConnector, fileServer, options, DEFAULT_OPTIONS)


    def train(self, stateDict, data, updateStateFun):
        '''
            Initializes a model based on the given stateDict and a data loader from the
            provided data and trains the model, taking into account the parameters speci-
            fied in the 'options' given to the class.
            Returns a serializable state dict of the resulting model.
        '''

        # initialize model
        model, labelclassMap = self.initializeModel(stateDict, data)

        # setup transform, data loader, dataset, optimizer, criterion
        transform = parse_transforms(self.options['train']['transform'])

        dataset = self.dataset_class(data, self.fileServer, labelclassMap,
                                transform, self.options['train']['ignore_unsure'],
                                **self.options['dataset']['kwargs']
                                )

        dataLoader = DataLoader(dataset,
                                collate_fn=collate,
                                **self.options['train']['dataLoader']['kwargs']
                                )

        optimizer_class = get_class_executable(self.options['train']['optim']['class'])
        optimizer = optimizer_class(params=model.parameters(), **self.options['train']['optim']['kwargs'])

        criterion_class = get_class_executable(self.options['train']['criterion']['class'])
        criterion = criterion_class(**self.options['train']['criterion']['kwargs'])

        # train model
        device = self.get_device()
        torch.manual_seed(self.options['general']['seed'])
        if 'cuda' in device:
            torch.cuda.manual_seed(self.options['general']['seed'])

        model.to(device)
        imgCount = 0
        for (img, labels, fVec, _) in tqdm(dataLoader):
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()
            pred = model(img)
            loss_value = criterion(pred, labels)
            loss_value.backward()
            optimizer.step()
            
            # update worker state
            imgCount += img.size(0)
            updateStateFun(state='PROGRESS', message='training', done=imgCount, total=len(dataLoader.dataset))

        # all done; return state dict as bytes
        return self.exportModelState(model)

    
    def inference(self, stateDict, data, updateStateFun):
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

        # setup transform, data loader, dataset, optimizer, criterion
        transform = parse_transforms(self.options['inference']['transform'])

        dataset = self.dataset_class(data, self.fileServer, labelclassMap,
                                transform, False,
                                **self.options['dataset']['kwargs']
                                )

        dataLoader = DataLoader(dataset,
                                collate_fn=collate,
                                **self.options['inference']['dataLoader']['kwargs']
                                )

        # perform inference
        device = self.get_device()
        response = {}
        model.to(device)
        imgCount = 0
        for (img, _, fVec, imgID) in tqdm(dataLoader):

            dataItem = img.to(device)
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
                            'logits': logits.cpu().numpy().tolist(),        #TODO: for AL criterion?
                            'confidence': torch.max(logits).item()
                        }
                    ]
                }
        
            # update worker state
            imgCount += len(imgID)
            updateStateFun(state='PROGRESS', message='predicting', done=imgCount, total=len(dataLoader.dataset))

        model.cpu()
        if 'cuda' in device:
            torch.cuda.empty_cache()

        return response