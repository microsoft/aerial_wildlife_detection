'''
    AIWorker model implementation based on PyTorch that supports fully and weakly supervised
    point detection models. The model must implement a function "getOutputSize(inputSize)",
    which returns a tuple (width, height) of int values denoting the predicted tensor size,
    given the inputSize tuple (input width, input height) as int values.

    In detail, the model accepts the following annotations:
        points: in this mode, the model is fully supervised and receives spatial ground truth.
                This ground truth is mapped to the grid cells the model predicts directly, and
                with respect to the annotations' label class values.
        labels: if non-spatial classification labels are given, the model is trained with weak
                supervision. In this mode, the model's predicted output is treated as a heatmap
                and trained with a custom binary loss that works as described in Kellenberger 
                et al., 2019 (see below). Naturally, this mode requires a fair balance between
                images that contain object(s) of a certain class, as well as images where the
                respective class is absent, so that the model can learn the appearance (and
                disappearance) of a certain class.
    
    For the weak supervision, see:
        Kellenberger, Benjamin, Diego Marcos, and Devis Tuia. "When a Few Clicks Make All the 
            Difference: Improving Weakly-Supervised Wildlife Detection in UAV Images." Procee-
            dings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops.
            2019.

    2019-20 Benjamin Kellenberger
'''

import io
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..genericPyTorchModel import GenericPyTorchModel
from .. import parse_transforms
from ._default_options import DEFAULT_OPTIONS
from ..functional._wsodPoints import encoder, collation

from util.helpers import get_class_executable, check_args


class PointModel(GenericPyTorchModel):

    def __init__(self, config, dbConnector, fileServer, options):
        super(PointModel, self).__init__(config, dbConnector, fileServer, options, DEFAULT_OPTIONS)
    

    def train(self, stateDict, data, updateStateFun):
        '''
            Initializes a model based on the given stateDict and a data loader from the
            provided data and trains the model, taking into account the parameters speci-
            fied in the 'options' given to the class.
            Trains the model with either mode (fully or weakly supervised), depending on
            the individual images' annotation types. This is handled by the dataset class.
            Returns a serializable state dict of the resulting model.
        '''

        # initialize model
        model, labelclassMap = self.initializeModel(stateDict, data)

        inputSize = tuple(self.options['general']['image_size'])
        targetSize = model.getOutputSize(inputSize)


        # setup transform, data loader, dataset, optimizer, criterion
        transform = parse_transforms(self.options['train']['transform'])

        dataset = self.dataset_class(data, self.fileServer, labelclassMap,
                                transform, self.options['train']['ignore_unsure'],
                                **self.options['dataset']['kwargs']
                                )
        
        dataEncoder = encoder.DataEncoder(len(labelclassMap.keys()))
        collator = collation.Collator(self.project, self.dbConnector, targetSize, dataEncoder)
        dataLoader = DataLoader(
            dataset=dataset,
            collate_fn=collator.collate_fn,
            **self.options['train']['dataLoader']['kwargs']
        )

        # optimizer
        optimizer_class = get_class_executable(self.options['train']['optim']['class'])
        optimizer = optimizer_class(params=model.parameters(), **self.options['train']['optim']['kwargs'])

        # loss criterion
        criterion_class = get_class_executable(self.options['train']['criterion']['class'])
        criterion = criterion_class(**self.options['train']['criterion']['kwargs'])

        # train model
        device = self.get_device()
        torch.manual_seed(self.options['general']['seed'])
        if 'cuda' in device:
            torch.cuda.manual_seed(self.options['general']['seed'])

        model.to(device)
        imgCount = 0
        for (img, locs_target, cls_images, fVec, _) in tqdm(dataLoader):
            img, locs_target, cls_images = img.to(device), \
                                                locs_target.to(device), \
                                                cls_images.to(device)
            
            optimizer.zero_grad()
            locs_pred = model(img)
            loss_value = criterion(locs_pred, locs_target, cls_images)
            loss_value.backward()
            optimizer.step()
            
            # update worker state
            imgCount += img.size(0)
            updateStateFun(state='PROGRESS', message='training', done=imgCount, total=len(dataLoader.dataset))

        # all done; return state dict as bytes
        return self.exportModelState(model)


    def inference(self, stateDict, data, updateStateFun):
        '''
            TODO
        '''

        # initialize model
        if stateDict is None:
            raise Exception('No trained model state found, but required for inference.')

        # read state dict from bytes
        model, labelclassMap = self.initializeModel(stateDict, data)

        inputSize = tuple(self.options['general']['image_size'])
        targetSize = model.getOutputSize(inputSize)

        # setup transform, data loader, dataset, optimizer, criterion
        transform = parse_transforms(self.options['inference']['transform'])

        dataset = self.dataset_class(data, self.fileServer, labelclassMap,
                                transform, False,
                                **self.options['dataset']['kwargs']
                                )

        dataEncoder = encoder.DataEncoder(len(labelclassMap.keys()))
        collator = collation.Collator(self.project, self.dbConnector, targetSize, dataEncoder)
        dataLoader = DataLoader(
            dataset=dataset,
            collate_fn=collator.collate_fn,
            **self.options['train']['dataLoader']['kwargs']
        )
        
        # perform inference
        device = self.get_device()
        response = {}
        model.to(device)
        imgCount = 0
        for (img, _, _, fVec, imgID) in tqdm(dataLoader):

            dataItem = img.to(device)
            with torch.no_grad():
                pred_batch = model(dataItem)
            
            # decode and append to dict
            for i in range(len(imgID)):
                pred_points, pred_labels, pred_confs = dataEncoder.decode(pred_batch[i,...].squeeze(),
                                                        min_conf=0.1, nms_dist=2)   #TODO
                
                predictions = []
                for p in range(pred_points.size(0)):
                    predictions.append({
                        'x': pred_points[p,0].item(),
                        'y': pred_points[p,1].item(),
                        'label': dataset.labelclassMap_inv[pred_labels[p].item()],
                        'logits': pred_confs[p,:].cpu().numpy().tolist(),
                        'confidence': torch.max(pred_confs[p,:]).item()
                    })
            
            response[imgID[i]] = {
                'predictions': predictions,
                #TODO: exception if fVec is not torch tensor: 'fVec': io.BytesIO(fVec.numpy().astype(np.float32)).getvalue()
            }
        
            # update worker state
            imgCount += len(imgID)
            updateStateFun(state='PROGRESS', message='predicting', done=imgCount, total=len(dataLoader.dataset))

        model.cpu()
        if 'cuda' in device:
            torch.cuda.empty_cache()

        return response