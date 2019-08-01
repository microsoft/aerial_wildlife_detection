import io
from tqdm import tqdm
from celery import current_task
import torch
from torch.utils.data import DataLoader
# from torchvision import transforms as tr
# import numpy as np
from ai.models import AIModel
from .. import parse_transforms, get_device

#TODO: clean up imports
from ai.models.pytorch.functional._retinanet import DEFAULT_OPTIONS
from ai.models.pytorch.functional._retinanet import collation, encoder, loss
from ai.models.pytorch.functional._retinanet.model import RetinaNet as Model
from ai.models.pytorch.functional.datasets.bboxDataset import BoundingBoxesDataset
from util.helpers import get_class_executable, check_args



class RetinaNet(AIModel):

    def __init__(self, config, dbConnector, fileServer, options):
        super(RetinaNet, self).__init__(config, dbConnector, fileServer, options)
        self.options = check_args(self.options, DEFAULT_OPTIONS)


    def train(self, stateDict, data):
        '''
            Initializes a model based on the given stateDict and a data loader from the
            provided data and trains the model, taking into account the parameters speci-
            fied in the 'options' given to the class.
            Returns a serializable state dict of the resulting model.
        '''

        # initialize model
        if stateDict is not None:
            stateDict = torch.load(io.BytesIO(stateDict), map_location=lambda storage, loc: storage)
            model = Model.loadFromStateDict(stateDict)
            
            # mapping labelclass (UUID) to index in model (number)
            labelclassMap = stateDict['labelclassMap']
        else:
            # create new label class map
            labelclassMap = {}
            for index, lcID in enumerate(data['labelClasses']):
                labelclassMap[lcID] = index
            self.options['model']['labelclassMap'] = labelclassMap

            # initialize a fresh model
            model = Model.loadFromStateDict(self.options['model']['kwargs'])


        # initialize data loader, dataset, transforms, optimizer, criterion
        inputSize = tuple(self.options['general']['image_size'])
        transform = parse_transforms(self.options['train']['transform'])
        # transforms = bboxTr.Compose([
        #     bboxTr.Resize(inputSize),
        #     bboxTr.RandomHorizontalFlip(p=0.5),
        #     bboxTr.DefaultTransform(tr.ColorJitter(0.25, 0.25, 0.25, 0.01)),
        #     bboxTr.DefaultTransform(tr.ToTensor()),
        #     bboxTr.DefaultTransform(tr.Normalize(mean=[0.485, 0.456, 0.406],
        #                                         std=[0.229, 0.224, 0.225]))
        # ])  #TODO: ditto, also write functional.pytorch util to compose transformations
        dataset = BoundingBoxesDataset(data=data,
                                    fileServer=self.fileServer,
                                    labelclassMap=labelclassMap,
                                    targetFormat='xyxy',
                                    transform=transform,
                                    ignoreUnsure=self.options['train']['ignore_unsure'])
        dataEncoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   #TODO: implement into options
        collator = collation.Collator(inputSize, dataEncoder)
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

        # mapping labelClass (UUID) to index in model (number)
        labelclassMap = stateDict['labelclassMap']

        # initialize data loader, dataset, transforms
        inputSize = tuple(self.options['general']['image_size'])
        transform = parse_transforms(self.options['inference']['transform'])
        # transforms = bboxTr.Compose([
        #     bboxTr.Resize(inputSize),
        #     bboxTr.DefaultTransform(tr.ToTensor()),
        #     bboxTr.DefaultTransform(tr.Normalize(mean=[0.485, 0.456, 0.406],
        #                                         std=[0.229, 0.224, 0.225]))
        # ])  #TODO: ditto, also write functional.pytorch util to compose transformations

        
        dataset = BoundingBoxesDataset(data=data,
                                    fileServer=self.fileServer,
                                    labelclassMap=labelclassMap,
                                    transform=transform)
        dataEncoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   #TODO: ditto
        collator = collation.Collator(inputSize, dataEncoder)
        dataLoader = DataLoader(
            dataset=dataset,
            collate_fn=collator.collate_fn,
            **self.options['inference']['dataLoader']['kwargs']
        )

        # perform inference
        response = {}
        device = get_device(self.options)
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
                bboxes_pred_batch, labels_pred_batch = model(dataItem, False)   #TODO: isFeatureVector
                bboxes_pred_batch, labels_pred_batch, confs_pred_batch = dataEncoder.decode(bboxes_pred_batch.squeeze(0).cpu(),
                                    labels_pred_batch.squeeze(0).cpu(),
                                    (inputSize[1],inputSize[0],),
                                    cls_thresh=0.1, nms_thresh=0.1,
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


# #TODO
# if __name__ == '__main__':
#     import os

#     os.environ['AIDE_CONFIG_PATH'] = 'settings_windowCropping.ini'
#     from util.configDef import Config
#     from modules.Database.app import Database
#     from modules.AIWorker.backend.worker.fileserver import FileServer
#     config = Config()
#     dbConnector = Database(config)
#     fileServer = FileServer(config)

#     rn = RetinaNet(config, dbConnector, fileServer, None)


#     # do inference on unlabeled
#     def __load_model_state(config, dbConnector):
#         # load model state from database
#         sql = '''
#             SELECT query.statedict FROM (
#                 SELECT statedict, timecreated
#                 FROM {schema}.cnnstate
#                 ORDER BY timecreated ASC NULLS LAST
#                 LIMIT 1
#             ) AS query;
#         '''.format(schema=config.getProperty('Database', 'schema'))
#         stateDict = dbConnector.execute(sql, None, numReturn=1)     #TODO: issues Celery warning if no state dict found
#         if not len(stateDict):
#             # force creation of new model
#             stateDict = None
        
#         else:
#             # extract
#             stateDict = stateDict[0]['statedict']

#         return stateDict
#     stateDict = __load_model_state(config, dbConnector)


#     #TODO TODO
#     from constants.dbFieldNames import FieldNames_annotation
#     def __load_metadata(config, dbConnector, imageIDs, loadAnnotations):
#         schema = config.getProperty('Database', 'schema')

#         # prepare
#         meta = {}

#         # label names
#         labels = {}
#         sql = 'SELECT * FROM {schema}.labelclass;'.format(schema=schema)
#         result = dbConnector.execute(sql, None, 'all')
#         for r in result:
#             labels[r['id']] = r     #TODO: make more elegant?
#         meta['labelClasses'] = labels

#         # image data
#         imageMeta = {}
#         sql = 'SELECT * FROM {schema}.image WHERE id IN %s'.format(schema=schema)
#         result = dbConnector.execute(sql, (tuple(imageIDs),), 'all')
#         for r in result:
#             imageMeta[r['id']] = r  #TODO: make more elegant?


#         # annotations
#         if loadAnnotations:
#             fieldNames = list(getattr(FieldNames_annotation, config.getProperty('Project', 'predictionType')).value)
#             sql = '''
#                 SELECT id AS annotationID, image, {fieldNames} FROM {schema}.annotation AS anno
#                 WHERE image IN %s;
#             '''.format(schema=schema, fieldNames=','.join(fieldNames))
#             result = dbConnector.execute(sql, (tuple(imageIDs),), 'all')
#             for r in result:
#                 if not 'annotations' in imageMeta[r['image']]:
#                     imageMeta[r['image']]['annotations'] = []
#                 imageMeta[r['image']]['annotations'].append(r)      #TODO: make more elegant?
#         meta['images'] = imageMeta

#         return meta

#     sql = '''SELECT image FROM aerialelephants_wc.image_user WHERE viewcount > 0 LIMIT 4096''' 
#     imageIDs = dbConnector.execute(sql, None, 4096)
#     imageIDs = [i['image'] for i in imageIDs]

#     data = __load_metadata(config, dbConnector, imageIDs, False)

#     # stateDict = rn.train(stateDict, data)

#     print('debug')

#     rn.inference(stateDict, data)