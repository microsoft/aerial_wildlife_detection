#TODO: define function shells in an abstract superclass (with documentation) and a template.

import io
import importlib
from tqdm import tqdm
from celery import current_task
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms as tr
from ai.models.pytorch.functional._retinanet import DEFAULT_OPTIONS
from ai.models.pytorch.functional._retinanet import collation, encoder, loss
from ai.models.pytorch.functional._retinanet.model import RetinaNet as Model
import ai.models.pytorch.functional._util.bboxTransforms as bboxTr
from ai.models.pytorch.functional.datasets.bboxDataset import BoundingBoxDataset
from util.helpers import get_class_executable



class RetinaNet:

    def __init__(self, config, dbConnector, fileServer, options):
        self.config = config
        self.dbConnector = dbConnector
        self.fileServer = fileServer
        self._parse_options(options)


    def _parse_options(self, options):
        '''
            Check for presence of required values and add defaults if not there.
        '''
        def __check_args(options, defaultOptions):
            if not isinstance(defaultOptions, dict):
                return options
            for key in defaultOptions.keys():
                if not key in options:
                    options[key] = defaultOptions[key]
                options[key] = __check_args(options[key], defaultOptions[key])
            return options
        if options is None or not isinstance(options, dict):
            self.options = DEFAULT_OPTIONS
        else:
            self.options = __check_args(options, DEFAULT_OPTIONS)


    def _get_device(self):
        device = self.options['general']['device']
        if 'cuda' in device and not torch.cuda.is_available():
            device = 'cpu'
        return device



    def train(self, stateDict, data):
        '''
            Initializes a model based on the given stateDict and a data loader from the
            provided data and trains the model, taking into account the parameters speci-
            fied in the 'options' given to the class.
            Returns a serializable state dict of the resulting model.
        '''

        # initialize model
        if stateDict is not None:
            model = Model.loadFromStateDict(stateDict)
        else:
            # initialize a fresh model
            self.options['model']['numClasses'] = len(data['labelClasses'])    #TODO
            model = Model.loadFromStateDict(self.options['model'])


        # initialize data loader, dataset, transforms, optimizer, criterion
        inputSize = tuple(self.options['general']['image_size'])
        transforms = bboxTr.Compose([
            bboxTr.Resize(inputSize),
            bboxTr.RandomHorizontalFlip(p=0.5),
            bboxTr.DefaultTransform(tr.ColorJitter(0.25, 0.25, 0.25, 0.01)),
            bboxTr.DefaultTransform(tr.ToTensor()),
            bboxTr.DefaultTransform(tr.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]))
        ])  #TODO: ditto, also write functional.pytorch util to compose transformations
        dataset = BoundingBoxDataset(data,
                                    self.fileServer,
                                    targetFormat='xyxy',
                                    transform=transforms,
                                    ignoreUnsure=self.options['train']['ignore_unsure'])
        dataEncoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   #TODO: implement into options
        collator = collation.Collator(inputSize, dataEncoder)
        dataLoader = DataLoader(
            dataset,
            batch_size=self.options['train']['batch_size'],
            shuffle=True,
            collate_fn=collator.collate_fn
        )

        # optimizer
        optimizer_class = get_class_executable(self.options['train']['optim']['class'])
        optimizer = optimizer_class(params=model.parameters(), **self.options['train']['optim']['kwargs'])        

        # loss criterion
        criterion_class = get_class_executable(self.options['train']['criterion']['class'])
        criterion = criterion_class(**self.options['train']['criterion']['kwargs'])

        # train model
        #TODO: outsource into dedicated function; set GPU, set random seed, etc.
        device = self._get_device()
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
        if 'cuda' in device:
            torch.cuda.empty_cache()
        model.cpu()

        bio = io.BytesIO()
        torch.save(model.getStateDict(), bio)
        return bio.getvalue()


    def average_model_states(self, stateDicts):
        '''
            TODO
        '''

        # read state dicts from bytes
        for s in range(len(stateDicts)):
            stateDict = io.BytesIO(stateDicts[s])
            stateDicts[s] = torch.load(stateDict, map_location=lambda storage, loc: storage)

        average_states = Model.averageStateDicts(stateDicts)

        # all done; return state dict as bytes
        bio = io.BytesIO()
        torch.save(average_states, bio)
        return bio.getvalue()

    
    def inference(self, stateDict, data):
        '''
            TODO
        '''

        # initialize model
        if stateDict is None:
            raise Exception('No trained model state found, but required for inference.')

        # read state dict from bytes
        stateDict = io.BytesIO(stateDict)
        stateDict = torch.load(stateDict, map_location=lambda storage, loc: storage)
        
        model = Model.loadFromStateDict(stateDict)


        # initialize data loader, dataset, transforms
        inputSize = tuple(self.options['general']['image_size'])
        transforms = bboxTr.Compose([
            bboxTr.Resize(inputSize),
            bboxTr.DefaultTransform(tr.ToTensor()),
            bboxTr.DefaultTransform(tr.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]))
        ])  #TODO: ditto, also write functional.pytorch util to compose transformations

        #TODO
        # dataType = self.options['general']['dataType'].lower()      # 'image' or 'featureVector'

        
        dataset = BoundingBoxDataset(data=data,
                                    fileServer=self.fileServer,
                                    transform=transforms)  #TODO: ditto
        dataEncoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   #TODO: ditto
        collator = collation.Collator(inputSize, dataEncoder)
        dataLoader = DataLoader(
            dataset,
            batch_size=self.options['inference']['batch_size'],     #TODO: at the moment the RetinaNet decoder doesn't support batch sizes > 1...
            shuffle=False,
            collate_fn=collator.collate_fn
        )

        # perform inference
        response = {}
        device = self._get_device()
        model.to(device)
        imgCount = 0
        for (img, _, _, fVec, imgID) in tqdm(dataLoader):

            # # BIG FAT TODO: BATCH SIZE... >:{
            # if img is not None:
            #     dataItem = img.to(device)
            #     isFeatureVector = False
            # else:
            #     dataItem = fVec.to(device)
            #     isFeatureVector = True
            dataItem = img.to(device)

            with torch.no_grad():
                bboxes_pred, labels_pred = model(dataItem, False)   #TODO: isFeatureVector
                bboxes_pred, labels_pred, confs_pred = dataEncoder.decode(bboxes_pred.squeeze(0).cpu(),
                                    labels_pred.squeeze(0).cpu(),
                                    (inputSize[1],inputSize[0],),
                                    cls_thresh=0.1, nms_thresh=0.1,
                                    return_conf=True)       #TODO: ditto

                if bboxes_pred.dim() == 2:
                    bboxes_pred = bboxes_pred.unsqueeze(0)
                    labels_pred = labels_pred.unsqueeze(0)
                    confs_pred = confs_pred.unsqueeze(0)

                for i in range(len(imgID)):
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
                                'label': dataset.classdef_inv[label.item()],
                                'logits': list(logits.numpy()),        #TODO: for AL criterion?
                                'confidence': torch.max(logits).item()
                            })
                    
                    response[imgID[i]] = {
                        'predictions': predictions,
                        'fVec': fVec        #TODO: maybe unnecessary (if fVec already there), cast to byte array (io.BytesIO(fVec.numpy().astype(np.float32)).getvalue())
                    }

            # update worker state   TODO
            imgCount += len(imgID)
            current_task.update_state(state='PROGRESS', meta={'done': imgCount, 'total': len(dataLoader.dataset), 'message': 'predicting'})

        model.cpu()
        if 'cuda' in device:
            torch.cuda.empty_cache()

        return response