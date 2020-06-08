'''
    RetinaNet trainer for PyTorch.

    2019-20 Benjamin Kellenberger
'''

import io
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ..genericPyTorchModel import GenericPyTorchModel
from .. import parse_transforms

from ..functional._retinanet import DEFAULT_OPTIONS, collation, encoder, loss
from ..functional._retinanet.model import RetinaNet as Model
from ..functional.datasets.bboxDataset import BoundingBoxesDataset
from util.helpers import get_class_executable
from util import optionsHelper


'''
    Map between new (GUI-enhanced) and old options JSON format fields.
    In the new format, all options are rooted under "options".
'''
OPTIONS_MAPPING = {
    'general.device.value': 'general.device',
    'general.seed.value': 'general.seed',
    'model.backbone.value': 'model.kwargs.backbone',
    'model.pretrained.value': 'model.kwargs.pretrained',
    'model.out_planes.value': 'model.kwargs.out_planes',
    'model.convertToInstanceNorm.value': 'model.kwargs.convertToInstanceNorm',
    'train.dataLoader.shuffle.value': 'train.dataLoader.kwargs.shuffle',
    'train.dataLoader.batch_size.value': 'train.dataLoader.kwargs.batch_size',
    'train.criterion.gamma.value': 'train.criterion.kwargs.gamma',
    'train.criterion.alpha.value': 'train.criterion.kwargs.alpha',
    'train.criterion.background_weight.value': 'train.criterion.kwargs.background_weight',
    'train.ignore_unsure': 'train.ignore_unsure',
    'inference.dataLoader.batch_size.value': 'inference.dataLoader.kwargs.batch_size'

    # optimizer and transforms are treated separately
}


class RetinaNet(GenericPyTorchModel):

    model_class = Model

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(RetinaNet, self).__init__(project, config, dbConnector, fileServer, options)

        self.model_class = Model


    ''' Model options parsing and verification functionalities '''

    @staticmethod
    def getDefaultOptions():
        jsonFile = 'config/ai/model/pytorch/boundingBoxes/retinanet.json'
        try:
            # try to load defaults from JSON file first
            options = json.load(open(jsonFile, 'r'))
        except Exception as e:
            # error; fall back to built-in defaults
            print(f'Error reading default RetinaNet options file "{jsonFile}" (message: "{str(e)}"), falling back to built-in options.')
            options = DEFAULT_OPTIONS
        
        # expand options
        options = optionsHelper.substitute_definitions(options)

        return options


    @staticmethod
    def _convertOldOptions(options, defaults):
        '''
            Receives options in the previous JSON encoding
            and converts them to the new GUI-enhanced scheme.
            Returns the new, converted options accordingly.
        '''
        newOptions = defaults.copy()

        warnings = []
        
        # update defaults key by key
        for key in OPTIONS_MAPPING.keys():
            newTokens = ['options']
            newTokens.extend(key.split('.'))
            oldTokens = OPTIONS_MAPPING[key].split('.')
            oldValue = optionsHelper.get_hierarchical_value(options, oldTokens, None)
            if oldValue is None:
                warnings.append(f'Value for options "{key}" could not be found in given options (expected location: "{OPTIONS_MAPPING[key]}").')
            else:
                optionsHelper.set_hierarchical_value(newOptions, newTokens, oldValue)

        # take special care of the optimizer: try all possible values (only the ones present will be retained)
        currentOptimType = options['train']['optim']['class']
        optionsHelper.set_hierarchical_value(newOptions, ('train','optim','value'), currentOptimType)
        optionsHelper.update_hierarchical_value(options, newOptions, ('train','optim','options',currentOptimType,'lr','value'), ('train', 'optim', 'kwargs', 'lr'))
        optionsHelper.update_hierarchical_value(options, newOptions, ('train','optim','options',currentOptimType,'weight_decay','value'), ('train', 'optim', 'kwargs', 'weight_decay'))
        optionsHelper.update_hierarchical_value(options, newOptions, ('train','optim','options',currentOptimType,'momentum','value'), ('train', 'optim', 'kwargs', 'momentum'))
        optionsHelper.update_hierarchical_value(options, newOptions, ('train','optim','options',currentOptimType,'alpha','value'), ('train', 'optim', 'kwargs', 'alpha'))
        optionsHelper.update_hierarchical_value(options, newOptions, ('train','optim','options',currentOptimType,'centered','value'), ('train', 'optim', 'kwargs', 'centered'))
        optionsHelper.update_hierarchical_value(options, newOptions, ('train','optim','options',currentOptimType,'dampening','value'), ('train', 'optim', 'kwargs', 'dampening'))
        optionsHelper.update_hierarchical_value(options, newOptions, ('train','optim','options',currentOptimType,'nesterov','value'), ('train', 'optim', 'kwargs', 'nesterov'))

        # also take special care of the transforms
        def _update_transforms(currentTransforms):
            newTransforms = []
            for tr in currentTr_train:
                # get from template definition and then replace values
                trClass = tr['class']
                if trClass not in newOptions['defs']['transform']:
                    warnings.append(f'Transform "{trClass}" is not defined in the new scheme and will be substituted appropriately.')
                    continue
                newTr = newOptions['defs']['transform'][trClass]
                for kw in tr['kwargs'].keys():
                    if kw == 'size':
                        newTr['width']['value'] = tr['kwargs']['size'][0]
                        newTr['height']['value'] = tr['kwargs']['size'][1]
                    elif kw in ('brightness', 'contrast', 'saturation', 'hue'):
                        newTr[kw]['minV']['value'] = 0
                        newTr[kw]['maxV']['value'] = tr['kwargs'][kw]
                        warnings.append(f'{kw} values of transforms have been set as maximums (min: 0).')
                    elif kw in ('mean', 'std'):
                        newTr['mean']['r'] = tr['kwargs'][kw][0]
                        newTr['mean']['g'] = tr['kwargs'][kw][1]
                        newTr['mean']['b'] = tr['kwargs'][kw][2]
                    elif kw in newTr:
                        newTr[kw]['value'] = tr['kwargs'][kw]
                newTransforms.append(newTr)
            return newTransforms

        currentTr_train = options['train']['transform']['kwargs']['transforms']
        newTr_train = _update_transforms(currentTr_train)
        newOptions['options']['train']['transform']['value'] = newTr_train

        currentTr_inference = options['inference']['transform']['kwargs']['transforms']
        newTr_inference = _update_transforms(currentTr_inference)
        newOptions['options']['inference']['transform']['value'] = newTr_inference

        print('Old RetinaNet options successfully converted to new format.')
        return newOptions, warnings
        

    @staticmethod
    def _verify_transforms(transforms, allowGeometric=True):
        warnings, errors = [], []
        transforms_PIL_new, transforms_tensor_new = [], []
        currentInputType = None    # to keep track of transform order
        for tr in transforms:
            if isinstance(tr, str):
                # only an ID provided; encapsulate
                warnings.append(f'Using default arguments for transform "{tr}"')
                tr = {
                    'id': tr
                }
            trID = tr['id']
            trName = (tr['name'] if 'name' in tr else trID)
                
            if trID == 'ai.models.pytorch.boundingBoxes.DefaultTransform':
                if 'transform' in tr:
                    newTr, newWarn, newErr = RetinaNet._verify_transforms(
                                                            [tr['transform']], allowGeometric)
                    transforms_PIL_new.extend(newTr)    #TODO: Compose could contain mixed transforms
                    warnings.extend(newWarn)
                    errors.extend(newErr)
                else:
                    warnings.append(f'Default transform "{trName}" contains no sub-transform and will be skipped.')

            elif trID == 'ai.models.pytorch.boundingBoxes.Compose':
                if 'transforms' in tr:
                    newTr, newWarn, newErr = RetinaNet._verify_transforms(
                                                        tr['transforms'], allowGeometric)
                    transforms_PIL_new.extend(newTr)    #TODO: Compose could contain mixed transforms
                    warnings.extend(newWarn)
                    errors.extend(newErr)
                else:
                    warnings.append(f'Compose transform "{trName}" contains no sub-transforms and will be skipped.')

            if trID in (
                'torchvision.transforms.Normalize',
                'torchvision.transforms.RandomErasing'
            ):
                # transforms on torch.tensor; these come at the end
                transforms_tensor_new.append({
                    'id': 'ai.models.pytorch.boundingBoxes.DefaultTransform',
                    'transform': tr
                })
                if currentInputType is not None and currentInputType != 'tensor':
                    warnings.append(f'Transform "{trName}" operates on Torch.tensor, but current input is PIL.Image. Transforms might be reordered.')
                currentInputType = 'tensor'

            elif trID in (
                'ai.models.pytorch.boundingBoxes.RandomHorizontalFlip',
                'ai.models.pytorch.boundingBoxes.RandomFlip'
            ):
                # geometric transforms on PIL.Image
                if not allowGeometric:
                    warnings.append(f'Transform "{trName}" modifies the image geometrically, which is not allowed here. The transform is being skipped.')
                    continue
                transforms_PIL_new.append(tr)
                if currentInputType is not None and currentInputType != 'image':
                    warnings.append(f'Transform "{trName}" operates on PIL images, but current input is Torch.tensor. Transforms might be reordered.')
                    currentInputType = 'image'
            
            elif trID in (
                'ai.models.pytorch.boundingBoxes.Resize',
                'torchvision.transforms.ColorJitter',
                'torchvision.transforms.Grayscale',
                'torchvision.transforms.RandomGrayscale'
            ):
                # non-geometric (+ always allowed resize) transforms on PIL.Image
                transforms_PIL_new.append({
                    'id': 'ai.models.pytorch.boundingBoxes.DefaultTransform',
                    'transform': tr
                })
                if currentInputType is not None and currentInputType != 'image':
                    warnings.append(f'Transform "{trName}" operates on PIL images, but current input is Torch.tensor. Transforms might be reordered.')
                    currentInputType = None     # reset

            elif trID in (
                'ai.models.pytorch.boundingBoxes.RandomClip',
                'ai.models.pytorch.boundingBoxes.RandomSizedClip'
            ):
                # transforms that work on both PIL.Image and torch.tensor
                if currentInputType == 'tensor':
                    transforms_tensor_new.append(tr)
                else:
                    transforms_PIL_new.append(tr)

            else:
                # unsupported transform
                warnings.append(f'Transform "{trName}" is not a recognized option and will be skipped.')

        # assemble transforms
        transforms_out = transforms_PIL_new

        # insert a ToTensor operation at the right location
        transforms_out.append({
            'id': 'ai.models.pytorch.boundingBoxes.DefaultTransform',
            'transform': 'torchvision.transforms.ToTensor'
        })
        transforms_out.extend(transforms_tensor_new)
        return transforms_out, warnings, errors


    @staticmethod
    def verifyOptions(options):
        # get default options to compare to
        defaultOptions = RetinaNet.getDefaultOptions()

        # updated options with modifications made
        if options is None:
            updatedOptions = defaultOptions.copy()
        else:
            if not isinstance(options, dict):
                try:
                    options = json.loads(options)
                except Exception as e:
                    return {
                        'valid': False,
                        'warnings': [],
                        'errors': [
                            f'Options are not in valid JSON format (message: "{str(e)}").'
                        ]
                    }
            updatedOptions = options.copy()

        result = {
            'valid': True,
            'warnings': [],
            'errors': []
        }

        if not 'defs' in updatedOptions:
            # old version (without GUI formatting): convert first
            updatedOptions, warnings = RetinaNet._convertOldOptions(updatedOptions, defaultOptions)
            result['warnings'].append('Options have been converted to new format.')
            result['warnings'].extend(warnings)

        # flatten and fill globals
        updatedOptions = optionsHelper.substitute_definitions(updatedOptions)

        # do the verification
        #TODO: verify rest

        # verify transforms
        transforms_train = updatedOptions['options']['train']['transform']['value']
        transforms_train, w, e = RetinaNet._verify_transforms(transforms_train, True)
        result['warnings'].extend(w)
        result['errors'].extend(e)
        if transforms_train is None:
            result['valid'] = False
        else:
            updatedOptions['options']['train']['transform']['value'] = transforms_train

        transforms_inf = updatedOptions['options']['inference']['transform']['value']
        transforms_inf, w, e = RetinaNet._verify_transforms(transforms_inf, False)
        result['warnings'].extend(w)
        result['errors'].extend(e)
        if transforms_inf is None:
            result['valid'] = False
        else:
            updatedOptions['options']['inference']['transform']['value'] = transforms_inf

        if result['valid']:
            result['options'] = updatedOptions

        return result


    @staticmethod
    def _init_transform_instances(transform, imageSize):
        '''
            Receives a list of transform definition dicts (or names)
            that are to be applied in order (either during training or
            for inference), and creates class instances for all of them.
            Also prepends a "Resize" operation (with the given image size)
            as well as a "DefaultTransform" with a "ToTensor" operation,
            to convert the image to a torch.Tensor instance.
            Returns a "Compose" transform with all the specified transforms
            in order.
        '''
        transforms = [{
            'id': 'ai.models.pytorch.boundingBoxes.Resize',
            'size': imageSize
        }]
        transforms.extend(transform)

        # check if "ToTensor" is needed
        hasToTensor = False
        for tr in transform:
            if tr['id'].endswith('DefaultTransform'):
                if (isinstance(tr['transform'], str) and tr['transform'].endswith('ToTensor')) or \
                    (isinstance(tr['transform'], dict) and tr['transform']['id'].endswith('ToTensor')):
                    hasToTensor = True
                    break

        if not hasToTensor:
            transforms.append({
                'id': 'ai.models.pytorch.boundingBoxes.DefaultTransform',
                'transform': {
                    'id': 'torchvision.transforms.ToTensor'
                }
            })
        transformsList = [{
            'id': 'ai.models.pytorch.boundingBoxes.Compose',
            'transforms': transforms
        }]
        transform_instances = GenericPyTorchModel.parseTransforms(transformsList)[0]
        return transform_instances




    ''' Model training and inference functionalities '''

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
        inputSize = (int(optionsHelper.get_hierarchical_value(self.options, ['options', 'general', 'imageSize', 'width', 'value'])),
                        int(optionsHelper.get_hierarchical_value(self.options, ['options', 'general', 'imageSize', 'height', 'value'])))
        
        transform = RetinaNet._init_transform_instances(
            optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'transform', 'value']),
            inputSize
        )

        dataset = BoundingBoxesDataset(data=data,
                                    fileServer=self.fileServer,
                                    labelclassMap=labelclassMap,
                                    targetFormat='xyxy',
                                    transform=transform,
                                    ignoreUnsure=optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'encoding', 'ignore_unsure', 'value'], fallback=False))

        dataEncoder = encoder.DataEncoder(
            minIoU_pos=optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'encoding', 'minIoU_pos', 'value'], fallback=0.5),
            maxIoU_neg=optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'encoding', 'maxIoU_neg', 'value'], fallback=0.4)
        )
        collator = collation.Collator(self.project, self.dbConnector, (inputSize[1], inputSize[0],), dataEncoder)
        dataLoader = DataLoader(
            dataset=dataset,
            collate_fn=collator.collate_fn,
            shuffle=optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'dataLoader', 'shuffle', 'value'], fallback=True)
        )

        # optimizer
        optimArgs = optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'optim', 'value'], None)
        optimArgs_out = {}
        optimClass = get_class_executable(optimArgs['id'])
        for key in optimArgs.keys():
            if key not in optionsHelper.RESERVED_KEYWORDS:
                optimArgs_out[key] = optionsHelper.get_hierarchical_value(optimArgs[key], ['value'])
        optimizer = optimClass(params=model.parameters(), **optimArgs_out)

        # loss criterion
        critArgs = optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'criterion'], None)
        critArgs_out = {}
        for key in critArgs.keys():
            if key not in optionsHelper.RESERVED_KEYWORDS:
                critArgs_out[key] = optionsHelper.get_hierarchical_value(critArgs[key], ['value'])
        criterion = loss.FocalLoss(**critArgs_out)

        # train model
        device = self.get_device()
        seed = int(optionsHelper.get_hierarchical_value(self.options, ['options', 'general', 'seed', 'value'], fallback=0))
        torch.manual_seed(seed)
        if 'cuda' in device:
            torch.cuda.manual_seed(seed)
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
            updateStateFun(state='PROGRESS', message='training', done=imgCount, total=len(dataLoader.dataset))

        # all done; return state dict as bytes
        return self.exportModelState(model)

    
    def inference(self, stateDict, data, updateStateFun):

        # initialize model
        if stateDict is None:
            raise Exception('No trained model state found, but required for inference.')

        # read state dict from bytes
        model, labelclassMap = self.initializeModel(stateDict, data)

        # initialize data loader, dataset, transforms
        inputSize = (int(optionsHelper.get_hierarchical_value(self.options, ['options', 'general', 'imageSize', 'width', 'value'])),
                        int(optionsHelper.get_hierarchical_value(self.options, ['options', 'general', 'imageSize', 'height', 'value'])))
        
        transform = RetinaNet._init_transform_instances(
            optionsHelper.get_hierarchical_value(self.options, ['options', 'inference', 'transform', 'value']),
            inputSize
        )
        
        dataset = BoundingBoxesDataset(data=data,
                                    fileServer=self.fileServer,
                                    labelclassMap=labelclassMap,
                                    transform=transform)
        dataEncoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   # IoUs don't matter for inference
        collator = collation.Collator(self.project, self.dbConnector, (inputSize[1], inputSize[0],), dataEncoder)
        dataLoader = DataLoader(
            dataset=dataset,
            collate_fn=collator.collate_fn,
            shuffle=False
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
                                    cls_thresh=optionsHelper.get_hierarchical_value(self.options, ['options', 'inference', 'encoding', 'cls_thresh', 'value'], fallback=0.1),
                                    nms_thresh=optionsHelper.get_hierarchical_value(self.options, ['options', 'inference', 'encoding', 'nms_thresh', 'value'], fallback=0.1),
                                    numPred_max=int(optionsHelper.get_hierarchical_value(self.options, ['options', 'inference', 'encoding', 'numPred_max', 'value'], fallback=128)),
                                    return_conf=True)

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
            updateStateFun(state='PROGRESS', message='predicting', done=imgCount, total=len(dataLoader.dataset))

        model.cpu()
        if 'cuda' in device:
            torch.cuda.empty_cache()

        return response