'''
    Base model trainer for models implemented towards the Detectron2 library
    (https://github.com/facebookresearch/detectron2).

    2020 Benjamin Kellenberger
'''

import os
import io
import copy
import json
from uuid import UUID
from tqdm import trange
import torch
from torch.nn.parallel import DistributedDataParallel
import detectron2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, MapDataset, build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.solver import build_lr_scheduler, build_optimizer
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage
from detectron2 import model_zoo

from ai.models import AIModel
from ._functional.dataset import getDetectron2Data
from ._functional.datasetMapper import Detectron2DatasetMapper
from ._functional.checkpointer import DetectionCheckpointerInMem
from util import optionsHelper


class GenericDetectron2Model(AIModel):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(GenericDetectron2Model, self).__init__(project, config, dbConnector, fileServer, options)

        # try to fill and substitute global definitions in JSON-enhanced options
        if isinstance(options, dict) and 'defs' in options:
            try:
                updatedOptions = optionsHelper.substitute_definitions(options.copy())
                self.options = updatedOptions
            except:
                # something went wrong; ignore
                pass
        
        # prepare Detectron2 configuration
        self.detectron2cfg = self._get_config()

        # write AIDE configuration values into Detectron2 config
        def _parse_aide_config(config):
            if isinstance(config, dict):
                for key in config.keys():
                    if key.startswith('DETECTRON2.'):
                        value = optionsHelper.get_hierarchical_value(config[key], ['value', 'id'])
                        if isinstance(value, list):
                            for v in range(len(value)):
                                value[v] = optionsHelper.get_hierarchical_value(value[v], ['value', 'id'])
                        
                        # copy over to Detectron2 configuration
                        tokens = key.split('.')
                        attr = self.detectron2cfg
                        for t in range(1,len(tokens)):    # skip "DETECTRON2" marker
                            if t == len(tokens) - 1:        # last element
                                setattr(attr, tokens[t], value)
                            else:
                                attr = getattr(attr, tokens[t])
                    else:
                        _parse_aide_config(config[key])

        _parse_aide_config(self.options)


    

    def _get_config(self):
        cfg = get_cfg()

        # augment and initialize Detectron2 cfg with selected model
        defaultConfig = optionsHelper.get_hierarchical_value(self.options, ['options', 'model', 'config', 'value', 'id'])
        if isinstance(defaultConfig, str):
            # try to load from Detectron2's model zoo
            try:
                configFile = model_zoo.get_config_file(defaultConfig)
            except:
                # not available; try to load locally instead
                configFile = os.path.join(os.getcwd(), 'ai/models/detectron2/_functional/configs', defaultConfig)
                
            cfg.merge_from_file(configFile)
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(defaultConfig)
        return cfg



    @classmethod
    def _load_default_options(cls, jsonFilePath, defaultOptions):
        try:
            # try to load defaults from JSON file first
            options = json.load(open(jsonFilePath, 'r'))
        except Exception as e:
            # error; fall back to built-in defaults
            print(f'Error reading default options file "{jsonFilePath}" (message: "{str(e)}"), falling back to built-in options.')
            options = defaultOptions
        
        # expand options
        options = optionsHelper.substitute_definitions(options)

        return options



    @classmethod
    def verifyOptions(cls, options):
        defaultOptions = cls.getDefaultOptions()
        if options is None:
            return {
                'valid': True,
                'options': defaultOptions
            }
        try:
            if isinstance(options, str):
                options = json.loads(options)
            options = optionsHelper.substitute_definitions(options)
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'Options are not in a proper format (message: {str(e)}).']
            }
        try:
            options, warnings, errors = optionsHelper.verify_options(options, autoCorrect=True)
            return {
                'valid': not len(errors),
                'warnings': warnings,
                'errors': errors,
                'options': options
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'An error occurred trying to verify options (message: {str(e)}).']
            }
    
    
    def initializeModel(self, stateDict, data):
        '''
            Loads Bytes object "stateDict" through torch and looks for a Detectron2
            config to initialize the model structure, optionally with pre-trained
            weights if available.
            Returns the model instance, the state dict, inter alia augmented with a
            'labelClassMap', defining the indexing between label classes and the model,
            and a list of new label classes (i.e., those that are present in the "data"
            dict, but not in the model definition).
            If the stateDict object is None, a new model and labelClassMap are crea-
            ted from the defaults.

            Note that this function does NOT modify the model w.r.t. the actually
            present label classes in the project. This functionality requires re-
            configuring the model's prediction output and depends on the model imple-
            mentation.
        '''
        # check if user has forced creating a new model
        forceNewModel = optionsHelper.get_hierarchical_value(self.options, ['options', 'model', 'force', 'value'], fallback=False)
        if isinstance(forceNewModel, bool) and forceNewModel:
            # print warning and reset flag
            print(f'[{self.project}] User has selected to force recreating a brand-new model.')
            optionsHelper.set_hierarchical_value(self.options, ['options', 'model', 'force', 'value'], False)

        # load state dict
        if stateDict is not None and not forceNewModel:
            stateDict = torch.load(io.BytesIO(stateDict), map_location=lambda storage, loc: storage)
        else:
            stateDict = {}

        # retrieve Detectron2 cfg
        if 'detectron2cfg' in stateDict and not forceNewModel:
            detectron2cfg = copy.deepcopy(stateDict['detectron2cfg'])
        else:
            detectron2cfg = copy.deepcopy(self.detectron2cfg)
            stateDict['detectron2cfg'] = self.detectron2cfg
        
        # check if CUDA is available; set to CPU temporarily if not
        if not torch.cuda.is_available():
            detectron2cfg.MODEL.DEVICE = 'cpu'

        # construct model and load state
        model = detectron2.modeling.build_model(detectron2cfg)
        checkpointer = DetectionCheckpointerInMem(model)
        if 'model_state' in stateDict and not forceNewModel:
            # trained weights available
            checkpointer.loadFromObject(stateDict)
        else:
            # fresh model; initialize from Detectron2 weights
            checkpointer.load(detectron2cfg.MODEL.WEIGHTS)
        
        # load or create labelclass map
        if 'labelclassMap' in stateDict and not forceNewModel:
            labelclassMap = stateDict['labelclassMap']
        else:
            labelclassMap = {}
        
            # add existing label classes; any non-UUID entry will be discarded during prediction
            try:
                pretrainedDataset = detectron2cfg.DATASETS.TRAIN
                if isinstance(pretrainedDataset, list) or isinstance(pretrainedDataset, tuple):
                    pretrainedDataset = pretrainedDataset[0]
                pretrainedMeta = MetadataCatalog.get(pretrainedDataset)
                for cID in pretrainedMeta.thing_dataset_id_to_contiguous_id.keys():
                    className = pretrainedMeta.thing_classes[pretrainedMeta.thing_dataset_id_to_contiguous_id[cID]]
                    labelclassMap[className] = cID - 1
            except:
                pass

        # check data for new label classes
        newClasses = []
        if len(labelclassMap):
            idx = max(labelclassMap.values()) + 1
        else:
            idx = 0
        for lcID in data['labelClasses']:
            if lcID not in labelclassMap:
                labelclassMap[lcID] = idx       #NOTE: we do not use the labelclass' serial 'idx', since this might contain gaps
                newClasses.append(lcID)
                idx += 1
        stateDict['labelclassMap'] = labelclassMap

        # parallelize model (if architecture supports it)   #TODO: try out whether/how well this works
        distributed = comm.get_world_size() > 1
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        return model, stateDict, newClasses

    

    def loadAndAdaptModel(self, stateDict, data):
        '''
            Loads a stateDict and initializes a model, just like the function
            "initializeModel". Unlike it, however, it also modifies the model
            to accept any new label classes for the current project. Since this
            procedure depends on the implementation, this function is just in
            the base class for compatibility, but not implemented.
        '''
        raise NotImplementedError('Only implemented by sub-models.')



    def exportModelState(self, stateDict, model):
        '''
            Retrieves a state dict from the model (e.g. after training) and converts it
            to a byte array that can be sent back to the AIWorker, and eventually the
            database.
            Also puts the model back on CPU and empties the CUDA cache (if available).
        '''
        # if 'cuda' in self.get_device():
        #     torch.cuda.empty_cache()
        model.cpu()

        stateDict['model_state'] = model.state_dict()

        bio = io.BytesIO()
        torch.save(stateDict, bio)

        return bio.getvalue()



    def initializeTransforms(self, mode='train', options=None):
        '''
            AIDE's Detectron2-compliant models all support the same transforms and thus
            can be initialized the same way. "mode" determines whether the transforms
            specified for 'train' or for 'inference' are to be initialized. If
            "options" contains a dict of AIDE model options, the transforms are to be
            initialized from there; otherwise the current class-specific ones are
            used.
        '''
        assert mode in ('train', 'inference'), 'Invalid transform mode specified'
        if isinstance(options, dict):
            opt = copy.deepcopy(options)
        else:
            opt = copy.deepcopy(self.options)
        opt = optionsHelper.substitute_definitions(opt)

        # parse transforms
        transforms = []
        transformOpts = optionsHelper.get_hierarchical_value(opt, ['options', mode, 'transform', 'value'])
        for tr in transformOpts:
            trClass = tr['id']
            args = optionsHelper.filter_reserved_children(tr, True)
            for a in args.keys():
                args[a] = optionsHelper.get_hierarchical_value(args[a], ['value'])
            
            # initialize
            transform = getattr(T, trClass)(**args)
            transforms.append(transform)
        return transforms


    
    def _build_optimizer(self, cfg, model):
        return build_optimizer(cfg, model)



    def _build_lr_scheduler(self, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)
        


    def train(self, stateDict, data, updateStateFun):
        '''
            Main training function.
        '''
        # initialize model
        model, stateDict = self.loadAndAdaptModel(stateDict, data)

        # wrap dataset for usage with Detectron2
        ignoreUnsure = optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'ignore_unsure', 'value'], fallback=True)
        transforms = self.initializeTransforms(mode='train')

        datasetMapper = Detectron2DatasetMapper(self.project, self.fileServer, transforms, True)
        dataLoader = build_detection_train_loader(
            dataset=getDetectron2Data(data, ignoreUnsure, self.detectron2cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS),
            mapper=datasetMapper,
            total_batch_size=self.detectron2cfg.SOLVER.IMS_PER_BATCH*comm.get_world_size(),      #TODO: verify
            aspect_ratio_grouping=True,
            num_workers=0
        )
        numImg = len(data['images'])
        
        # train
        model.train()
        optimizer = self._build_optimizer(stateDict['detectron2cfg'], model)
        scheduler = self._build_lr_scheduler(stateDict['detectron2cfg'], optimizer)
        imgCount = 0
        start_iter = 0      #TODO
        tbar = trange(numImg)
        with EventStorage(start_iter) as storage:
            for idx, batch in enumerate(dataLoader):
                storage.iter = idx  #TODO: start_iter
                loss_dict = model(batch)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), \
                    'Model produced Inf and/or NaN values; training was aborted. Try reducing the learning rate.'

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                scheduler.step()

                # update worker state
                tbar.update(1)
                imgCount += len(batch)
                updateStateFun(state='PROGRESS', message='training', done=imgCount, total=numImg)

            tbar.close()

        # all done; return state dict as bytes
        return self.exportModelState(stateDict, model)



    def average_model_states(self, stateDicts, updateStateFun):
        '''
            Receives a list of model states (as bytes) and calls the model's
            averaging function to return a single, unified model state.
        '''

        # read state dicts from bytes
        loadedStates = []
        for s in range(len(stateDicts)):
            stateDict = io.BytesIO(stateDicts[s])
            loadedStates.append(torch.load(stateDict, map_location=lambda storage, loc: storage))

        averagedWeights = loadedStates[0]['model_state']
        for key in averagedWeights.keys():
            for s in range(1,len(stateDicts)):
                averagedWeights[key] += loadedStates[s]['model_state'][key]
            averagedWeights[key] /= len(loadedStates)
        
        loadedStates[0]['model_state'] = averagedWeights

        # all done; return state dict as bytes
        bio = io.BytesIO()
        torch.save(loadedStates[0], bio)
        return bio.getvalue()



    def inference(self, stateDict, data, updateStateFun):
        '''
            Main inference function.
        '''
        # initialize model
        model, stateDict = self.loadAndAdaptModel(stateDict, data)

        # construct inverted labelclass map
        labelclassMap_inv = {}
        for key in stateDict['labelclassMap'].keys():
            index = stateDict['labelclassMap'][key]
            labelclassMap_inv[index] = key
        
        # wrap dataset for usage with Detectron2
        transforms = []
        transforms = self.initializeTransforms(mode='inference')

        datasetMapper = Detectron2DatasetMapper(self.project, self.fileServer, transforms, False)
        dataLoader = build_detection_test_loader(
            dataset=getDetectron2Data(data, False, False),
            mapper=datasetMapper,
            num_workers=stateDict['detectron2cfg'].DATALOADER.NUM_WORKERS
        )
        numImg = len(data['images'])

        # perform inference
        response = {}
        model.eval()
        imgCount = 0

        tbar = trange(numImg)
        with torch.no_grad():
            for idx, batch in enumerate(dataLoader):
                outputs = model(batch)
                outputs = outputs[0]['instances']

                labels = outputs.pred_classes
                scores = outputs.scores

                # export bboxes if predicted
                if hasattr(outputs, 'pred_boxes'):
                    bboxes = outputs.pred_boxes.tensor

                    # convert bboxes to relative XYWH format
                    bboxes[:,2] -= bboxes[:,0]
                    bboxes[:,3] -= bboxes[:,1]
                    bboxes[:,0] += bboxes[:,2]/2
                    bboxes[:,1] += bboxes[:,3]/2
                    bboxes[:,0] /= batch[0]['width']
                    bboxes[:,1] /= batch[0]['height']
                    bboxes[:,2] /= batch[0]['width']
                    bboxes[:,3] /= batch[0]['height']

                    predictions = []
                    for b in range(len(scores)):
                        label = labels[b].item()

                        if label not in labelclassMap_inv or not isinstance(labelclassMap_inv[label], UUID):
                            # prediction with invalid label (e.g. from pretrained model state)
                            continue

                        predictions.append({
                            'x': bboxes[b,0].item(),
                            'y': bboxes[b,1].item(),
                            'width': bboxes[b,2].item(),
                            'height': bboxes[b,3].item(),
                            'label': labelclassMap_inv[label],
                            'confidence': scores[b].item(),
                            #TODO: logits...
                        })
                
                #TODO: segmentation masks, image labels

                response[batch[0]['image_uuid']] = {
                    'predictions': predictions
                }

                # update worker state
                tbar.update(1)
                imgCount += len(batch)
                updateStateFun(state='PROGRESS', message='predicting', done=imgCount, total=numImg)

            tbar.close()

        return response