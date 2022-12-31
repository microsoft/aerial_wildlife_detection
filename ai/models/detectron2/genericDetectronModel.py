'''
    Base model trainer for models implemented towards the Detectron2 library
    (https://github.com/facebookresearch/detectron2).

    2020-22 Benjamin Kellenberger
'''

import os
import io
import copy
import json
from uuid import UUID
from tqdm import trange
from psycopg2 import sql
import torch
from torch.nn.parallel import DistributedDataParallel
import detectron2
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.solver import build_lr_scheduler, build_optimizer
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage
from detectron2 import model_zoo

from ai.models import AIModel
from util import optionsHelper
from ._functional.dataset import getDetectron2Data
from ._functional.collation import collate
from ._functional.datasetMapper import Detectron2DatasetMapper
from ._functional.checkpointer import DetectionCheckpointerInMem


class GenericDetectron2Model(AIModel):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(GenericDetectron2Model, self).__init__(project, config, dbConnector, fileServer, options)
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                # something went wrong; discard options #TODO
                options = None

        # try to fill and substitute global definitions in JSON-enhanced options
        if isinstance(options, dict):
            try:
                updatedOptions = optionsHelper.merge_options(self.getDefaultOptions(), options.copy())
                updatedOptions = optionsHelper.substitute_definitions(updatedOptions)
                self.options = updatedOptions
            except Exception:
                # something went wrong; ignore
                pass
        
        # verify options
        result = self.verifyOptions(self.options)
        if not result['valid']:
            print(f'[{self.project}] WARNING: provided options are invalid; replacing with defaults...')
            self.options = self.getDefaultOptions()

        # prepare Detectron2 configuration
        self.detectron2cfg = self._get_config()

        # # write AIDE configuration values into Detectron2 config (TODO: we now do this below)
        # self.detectron2cfg = self.parse_aide_config(self.options, self.detectron2cfg)



    @classmethod
    def parse_aide_config(cls, config, override_detectron2cfg=None):
        detectron2cfg = get_cfg()
        detectron2cfg.set_new_allowed(True)
        if override_detectron2cfg is not None:
            detectron2cfg.update(override_detectron2cfg)
        if isinstance(config, dict):
            for key in config.keys():
                if isinstance(key, str) and key.startswith('DETECTRON2.'):
                    value = optionsHelper.get_hierarchical_value(config[key], ['value', 'id'])
                    if isinstance(value, list):
                        for idx, val in enumerate(value):
                            value[idx] = optionsHelper.get_hierarchical_value(val, ['value', 'id'])

                    # copy over to Detectron2 configuration
                    tokens = key.split('.')
                    attr = detectron2cfg
                    for t in range(1,len(tokens)):    # skip "DETECTRON2" marker
                        if t == len(tokens) - 1:        # last element
                            setattr(attr, tokens[t], value)
                        else:
                            attr = getattr(attr, tokens[t])
                else:
                    GenericDetectron2Model.parse_aide_config(config[key], detectron2cfg)

        return detectron2cfg



    def _get_config(self):
        cfg = get_cfg()
        cfg.set_new_allowed(True)

        # augment and initialize Detectron2 cfg with selected model
        default_config = optionsHelper.get_hierarchical_value(self.options,
                                                    ['options', 'model', 'config', 'value', 'id'])
        if isinstance(default_config, str):
            # try to load from Detectron2's model zoo
            try:
                config_file = model_zoo.get_config_file(default_config)
            except Exception:
                # not available; try to load locally instead
                config_file = os.path.join(os.getcwd(),
                                        'ai/models/detectron2/_functional/configs', default_config)
                if not os.path.exists(config_file):
                    config_file = None

            if config_file is not None:
                cfg.merge_from_file(config_file)
            try:
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(default_config)
            except Exception:
                pass
        return cfg



    @classmethod
    def _load_default_options(cls, json_file_path, default_options):
        try:
            # try to load defaults from JSON file first
            with open(json_file_path, 'r', encoding='utf-8') as f_opts:
                options = json.load(f_opts)
        except Exception as exc:
            # error; fall back to built-in defaults
            print(f'Error reading default options file "{json_file_path}" ' + \
                f'(message: "{str(exc)}"), falling back to built-in options.')
            options = default_options

        # expand options
        options = optionsHelper.substitute_definitions(options)

        return options



    @classmethod
    def verifyOptions(cls, options):
        default_options = cls.getDefaultOptions()
        if options is None:
            return {
                'valid': True,
                'options': default_options
            }
        try:
            if isinstance(options, str):
                options = json.loads(options)
            options = optionsHelper.substitute_definitions(options)
        except Exception as exc:
            return {
                'valid': False,
                'errors': [f'Options are not in a proper format (message: {str(exc)}).']
            }
        try:
            # mandatory field: model config
            model_config = optionsHelper.get_hierarchical_value(options,
                                                        ['options', 'model', 'config', 'value'])
            if model_config is None:
                raise Exception('missing model type field in options.')

            opts, warnings, errors = optionsHelper.verify_options(options['options'],
                                                                        autoCorrect=True)
            options['options'] = opts
            return {
                'valid': len(errors) == 0,
                'warnings': warnings,
                'errors': errors,
                'options': options
            }
        except Exception as exc:
            return {
                'valid': False,
                'errors': [f'An error occurred trying to verify options (message: {str(exc)}).']
            }


    def loadModelWeights(self, model, state_dict, force_new_model):
        checkpointer = DetectionCheckpointerInMem(model)
        if 'model' in state_dict and not force_new_model:
            # trained weights available
            checkpointer.loadFromObject(state_dict)
        else:
            # fresh model; initialize from Detectron2 weights
            checkpointer.load(self.detectron2cfg.MODEL.WEIGHTS)



    def initializeModel(self, state_dict, data, revert_project_to_state_map=False):
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
        force_new_model = optionsHelper.get_hierarchical_value(self.options,
                                                            ['options', 'model', 'force', 'value'],
                                                            fallback=False)
        if isinstance(force_new_model, bool) and force_new_model:
            # print warning and reset flag
            print(f'[{self.project}] User has selected to force recreating a brand-new model.')
            optionsHelper.set_hierarchical_value(self.options,
                                                    ['options', 'model', 'force', 'value'],
                                                    False)
        else:
            force_new_model = False

        # load state dict
        if state_dict is not None and not force_new_model:
            if not isinstance(state_dict, dict):
                state_dict = torch.load(io.BytesIO(state_dict), map_location='cpu')
        else:
            state_dict = {}

        # retrieve Detectron2 cfg
        if 'detectron2cfg' in state_dict and not force_new_model:
            # medium priority: model overrides
            self.detectron2cfg.set_new_allowed(True)
            self.detectron2cfg.update(CfgNode(state_dict['detectron2cfg']))

        # top priority: AIDE config overrides
        self.detectron2cfg = self.parse_aide_config(self.options, self.detectron2cfg)

        state_dict['detectron2cfg'] = self.detectron2cfg

        # check if CUDA is available; set to CPU temporarily if not
        if not torch.cuda.is_available():
            self.detectron2cfg.MODEL.DEVICE = 'cpu'

        # construct model and load state
        model = detectron2.modeling.build_model(self.detectron2cfg)
        self.loadModelWeights(model, state_dict, force_new_model)

        # load or create labelclass map
        if 'labelclassMap' in state_dict and not force_new_model:
            labelclass_map = state_dict['labelclassMap']
        else:
            labelclass_map = {}

            # add existing label classes; any non-UUID entry will be discarded during prediction
            try:
                pretrained_dataset = self.detectron2cfg.DATASETS.TRAIN
                if isinstance(pretrained_dataset, list) or \
                    isinstance(pretrained_dataset, tuple) and \
                    len(pretrained_dataset) > 0:
                    pretrained_dataset = pretrained_dataset[0]
                pretrained_meta = MetadataCatalog.get(pretrained_dataset)
                for idx, class_id in enumerate(pretrained_meta.thing_classes):
                    labelclass_map[class_id] = class_id
            except Exception:
                pass

            if hasattr(model, 'names'):
                # YOLOv5 format; get names from there
                for idx, class_id in enumerate(model.names):
                    labelclass_map[class_id] = idx

        # check data for new label classes
        project_to_state_map = {}
        new_classes = []
        for labelclass_id, labelclass_meta in data['labelClasses'].items():
            if labelclass_id not in labelclass_map:
                # check if original label class got re-mapped
                lc_map_origin_id = labelclass_meta.get('labelclass_id_model', None)
                if lc_map_origin_id is None or lc_map_origin_id not in labelclass_map:
                    # no remapping done; class really is new
                    new_classes.append(labelclass_id)
                else:
                    # class has been re-mapped
                    if revert_project_to_state_map:
                        project_to_state_map[lc_map_origin_id] = labelclass_id
                    else:
                        project_to_state_map[labelclass_id] = lc_map_origin_id
        state_dict['labelclassMap'] = labelclass_map

        # parallelize model (if architecture supports it). TODO: try out whether/how well this works
        distributed = comm.get_world_size() > 1
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        return model, state_dict, new_classes, project_to_state_map



    def calculateClassCorrelations(self, stateDict, model, modelClasses, targetClasses, maxNumImagesPerClass=None):
        '''
            Determines the correlation between label classes predicted by the
            model and target annotations for a given set of "targetClasses".
            Does so through the following steps:
                1. Loads a number of images that contain at least one
                   annotation with label class in "targetClasses".
                2. Performs a forward pass with the existing model over
                   all images.
                3. Compares the predictions of the model with the target anno-
                   tations and calculates normalized "correlation" weights
                   between them. For bounding boxes, this is done through a
                   combination of class confidence scores and intersection-over-
                   union scores between the predicted and the ground truth box.

            Returns:
                - a list of torch.Tensors containing normalized weights for all
                  of the model's existing classes compared to each of the target
                  classes.
        '''
        raise NotImplementedError('Only implemented by sub-models.')

    

    def loadAndAdaptModel(self, stateDict, data, updateStateFun):
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

        stateDict['model'] = model.state_dict()

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

        #TODO: add a resize transform in any case

        transformOpts = optionsHelper.get_hierarchical_value(opt, ['options', mode, 'transform', 'value'])
        for tr in transformOpts:
            trClass = tr['id']
            args = optionsHelper.filter_reserved_children(tr, True)
            for a in args.keys():
                args[a] = optionsHelper.get_hierarchical_value(args[a], ['value'])
                if a == 'interp':
                    #TODO: ugly solution to convert interpolation methods
                    args[a] = int(args[a])

            # initialize
            transform = getattr(T, trClass)(**args)
            transforms.append(transform)
        return transforms



    def _get_labelclass_index_map(self, labelclassMap, reverse=False):
        '''
            For some annotation types (e.g., semantic segmentation),
            the labels stored in the ground truth are derived from
            the class' "idx" value as per database. This does not au-
            tomatically correspond to the true index, e.g. if a class
            gets removed. This function therefore creates a map from
            AIDE's official label class index to the model's
            labelclassMap.
            If "reverse" is True, a flipped map will be created that
            is to be used for inference (otherwise for training).
        '''
        indexMap = {}

        # get AIDE indices
        query = self.dbConnector.execute(sql.SQL('''
                SELECT id, idx FROM {}
            ''').format(sql.Identifier(self.project, 'labelclass')),
            None, 'all')
        for row in query:
            lcID = row['id']
            if lcID in labelclassMap:
                aideIdx = row['idx']
                modelIdx = labelclassMap[lcID]
                if reverse:
                    indexMap[modelIdx] = aideIdx
                else:
                    indexMap[aideIdx] = modelIdx
        return indexMap

    

    def _get_band_config(self, stateDict, data):
        '''
            Assembles a tuple of band indices that are to be provided to the
            model as input. For example, if the project contains images with six
            bands and the model provides a band configuration of (4,0,1), then a
            three-layer image with bands at given indices are fed to the model.
            Fallback A (model state dict does not provide a band config): the
            RGB values from the LabelUI frontend are selected (data >
            "render_config"). Fallback B (no render config is provided):
            defaults are assumed (three layer image with indices at 0, 1, 2 for
            Red, Green, Blue).
        '''
        # assemble band configuration
        bandConfig = (0,0,0)

        try:
            bandNames = data['band_config']
        except Exception:
            # no band names provided; assume RGB
            bandNames = ('Red', 'Green', 'Blue')

        #TODO: get band config from model state
        bandConfig = stateDict.get('bandConfig', None)

        if bandConfig is None:
            # fallback A: get from render_config
            try:
                renderConfig = data['render_config']
                try:
                    bandConfig = []
                    for key in ('Red', 'Green', 'Blue'):    #TODO: grayscale?
                        bandConfig.append(renderConfig['bands']['indices'][key])
                except Exception:
                    # fallback B: assume R-G-B
                    bandConfig = (
                        0,
                        min(1, len(bandNames)-1),
                        min(2, len(bandNames)-1)
                    )
                #TODO: for next version of AIDE: allow model to be expanded towards new bands
                
            except Exception:
                # fallback to default R-G-B
                bandConfig = (0, 1, 2)
        return bandConfig


    
    def _build_optimizer(self, cfg, model):
        return build_optimizer(cfg, model)



    def _build_lr_scheduler(self, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)



    def update_model(self, stateDict, data, updateStateFun):
        '''
            Updater function. Modifies the model to incorporate newly
            added label classes.
        '''
        # initialize model
        model, stateDict = self.loadAndAdaptModel(stateDict, data, updateStateFun)

        # all done; return state dict as bytes
        return self.exportModelState(stateDict, model)

        

    def train(self, stateDict, data, updateStateFun):
        '''
            Main training function.
        '''
        # initialize model
        model, stateDict, _, projectToStateMap = self.initializeModel(stateDict, data, False)

        # wrap dataset for usage with Detectron2
        ignoreUnsure = optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'ignore_unsure', 'value'], fallback=True)
        transforms = self.initializeTransforms(mode='train')
        indexMap = self._get_labelclass_index_map(stateDict['labelclassMap'], False)
        
        bandConfig = self._get_band_config(stateDict, data)
        # try:
        #     imageFormat = self.detectron2cfg.INPUT.FORMAT
        #     assert imageFormat.upper() in ('RGB', 'BGR')
        # except Exception:
        #     imageFormat = 'BGR'
        datasetMapper = Detectron2DatasetMapper(self.project, self.fileServer, transforms, True, bandConfig, classIndexMap=indexMap)
        dataLoader = build_detection_train_loader(
            dataset=getDetectron2Data(data, stateDict['labelclassMap'], projectToStateMap, ignoreUnsure, self.detectron2cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS),
            mapper=datasetMapper,
            total_batch_size=self.detectron2cfg.SOLVER.IMS_PER_BATCH*comm.get_world_size(),      #TODO: verify
            aspect_ratio_grouping=True,
            collate_fn=collate,
            num_workers=0
        )
        numImg = len(data['images'])
        
        # train
        model.train()
        optimizer = self._build_optimizer(self.detectron2cfg, model)
        scheduler = self._build_lr_scheduler(self.detectron2cfg, optimizer)
        imgCount = 0
        start_iter = 0      #TODO
        tbar = trange(numImg)
        dataLoaderIter = iter(dataLoader)
        with EventStorage(start_iter) as storage:
            for idx in range(numImg):
                batch = next(dataLoaderIter)
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

            stats = storage.latest()
            for key in stats:
                if isinstance(stats[key], tuple):
                    stats[key] = stats[key][0]
            tbar.close()

        # all done; return state dict as bytes and stats
        return self.exportModelState(stateDict, model), stats



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

        averagedWeights = loadedStates[0]['model']
        for key in averagedWeights.keys():
            for s in range(1,len(stateDicts)):
                averagedWeights[key] += loadedStates[s]['model'][key]
            averagedWeights[key] /= len(loadedStates)
        
        loadedStates[0]['model'] = averagedWeights

        # all done; return state dict as bytes
        bio = io.BytesIO()
        torch.save(loadedStates[0], bio)
        return bio.getvalue()



    def inference(self, stateDict, data, updateStateFun):
        '''
            Main inference function.
        '''
        # initialize model
        model, stateDict, _, stateToProjectMap = self.initializeModel(stateDict, data, True)

        # construct inverted labelclass map, taking model-to-project mapping into account with priority
        labelclassMap_inv = {}
        for key in stateDict['labelclassMap'].keys():
            if key in stateToProjectMap:
                target = stateToProjectMap[key]
            else:
                target = key
            if target in data['labelClasses']:
                index = stateDict['labelclassMap'][key]
                labelclassMap_inv[index] = target
        
        # wrap dataset for usage with Detectron2
        transforms = self.initializeTransforms(mode='inference')
        indexMap = self._get_labelclass_index_map(stateDict['labelclassMap'], True)

        bandConfig = self._get_band_config(stateDict, data)
        # try:
        #     imageFormat = self.detectron2cfg.INPUT.FORMAT
        #     assert imageFormat.upper() in ('RGB', 'BGR')
        # except Exception:
        #     imageFormat = 'BGR'
        datasetMapper = Detectron2DatasetMapper(self.project, self.fileServer, transforms, False, bandConfig)
        dataLoader = build_detection_test_loader(
            dataset=getDetectron2Data(data, stateDict['labelclassMap'], None, False, False),
            mapper=datasetMapper,
            collate_fn=collate,
            num_workers=0
        )
        numImg = len(data['images'])

        # perform inference
        response = {}
        model.eval()
        imgCount = 0

        tbar = trange(numImg)
        dataLoaderIter = iter(dataLoader)
        with torch.no_grad():
            for _ in range(numImg):
                batch = next(dataLoaderIter)
                outputs = model(batch)
                outputs = outputs[0]
                
                predictions = []

                if 'instances' in outputs:
                    outputs = outputs['instances']

                    labels = outputs.pred_classes.cpu().int()
                    scores = outputs.scores.cpu()

                    # export instance masks as polygons if predicted
                    if hasattr(outputs, 'pred_masks'):
                        pass    #TODO: convert masks to polygons

                    # export bboxes if predicted
                    if hasattr(outputs, 'pred_boxes'):
                        bboxes = outputs.pred_boxes.tensor.cpu()
                        
                        # convert bboxes to relative XYWH format; rescale if needed
                        scaleFactor = torch.tensor([
                            float(batch[0]['width']) / outputs.image_size[1],
                            float(batch[0]['height']) / outputs.image_size[0]
                        ]).repeat(2)
                        bboxes *= scaleFactor

                        bboxes[:,2] -= bboxes[:,0]
                        bboxes[:,3] -= bboxes[:,1]
                        bboxes[:,0] += bboxes[:,2]/2
                        bboxes[:,1] += bboxes[:,3]/2
                        bboxes[:,0] /= batch[0]['width']
                        bboxes[:,1] /= batch[0]['height']
                        bboxes[:,2] /= batch[0]['width']
                        bboxes[:,3] /= batch[0]['height']
                        
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
                
                elif 'sem_seg' in outputs:
                    outputs = outputs['sem_seg']
                    confidence, label = torch.max(outputs, 0)

                    # map back to AIDE indices
                    label_copy = label.clone()
                    for k, v in indexMap.items(): label_copy[label==k] = v
                    label = label_copy

                    predictions.append({
                        'label': label.cpu().numpy(),
                        'logits': outputs.cpu().numpy().tolist(),
                        'confidence': confidence.cpu().numpy()
                    })
                
                elif 'pred_label' in outputs:
                    labelIndex = outputs['pred_label'].item()
                    if labelIndex not in labelclassMap_inv:
                        # update worker state
                        tbar.update(1)
                        imgCount += len(batch)
                        updateStateFun(state='PROGRESS', message='predicting', done=imgCount, total=numImg)
                        continue
                    
                    predictions.append({
                        'label': labelclassMap_inv[labelIndex],
                        'logits': outputs['pred_logits'].cpu().numpy().tolist(),
                        'confidence': outputs['pred_conf'].item()
                    })

                #TODO: instance segmentation, etc.

                response[batch[0]['image_uuid']] = {
                    'predictions': predictions
                }

                # update worker state
                tbar.update(1)
                imgCount += len(batch)
                updateStateFun(state='PROGRESS', message='predicting', done=imgCount, total=numImg)

            tbar.close()

        return response