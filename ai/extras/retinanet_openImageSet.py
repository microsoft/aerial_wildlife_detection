'''
    RetinaNet model with the inference function replaced:
    Here, we do inference on full images, split into patches,
    instead of going through data already in the database.

    2019 Benjamin Kellenberger
'''

import os
import io
import glob
import re
import random
import numpy as np
from tqdm import tqdm
from celery import current_task
import torch
from torchvision import transforms as tr
from PIL import Image
import rawpy
from util.helpers import check_args
from ai.models.pytorch.detection.retinanet import RetinaNet
from ai.models.pytorch.functional._retinanet.model import RetinaNet as Model
from ai.models.pytorch.functional._retinanet import encoder
from ai.models.pytorch.functional._retinanet.utils import box_nms
from ai.extras._functional import tensorSharding, windowCropping


class RetinaNet_ois(RetinaNet):

    def __init__(self, config, dbConnector, fileServer, options):
        super(RetinaNet_ois, self).__init__(config, dbConnector, fileServer, options)

        # add contrib options
        defaultContribOptions = {
            'baseFolder_unlabeled': '/datadrive/hfaerialblobs/_images/',   # local folder to search for non-added images
            'load_raw_images': True,   # whether to take RAW files into account
            'inference_max_num_unlabeled': 32,
            'export_empty_patches': False,
            'stride': 0.65          # relative stride factor
        }
        if not 'contrib' in self.options:
            self.options['contrib'] = defaultContribOptions
        else:
            self.options['contrib'] = check_args(self.options['contrib'], defaultContribOptions)


        # parameters
        self.batchSize = self.options['inference']['batch_size']
        self.maxNumUnlabeled = self.options['contrib']['inference_max_num_unlabeled']
        self.patchSize = tuple(self.options['general']['image_size'])
        self.stride = self.options['contrib']['stride']
        self.encoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   #TODO: implement 

        self.windowCropper = windowCropping.WindowCropper(
            patchSize=self.patchSize, exportEmptyPatches=False,
            cropMode='windowCropping',
            searchStride=(10,10,),
            minBBoxArea=64, minBBoxAreaFrac=0.25  #TODO
        )

        self.baseFolder_unlabeled = self.options['contrib']['baseFolder_unlabeled']
        self.loadRaw = self.options['contrib']['load_raw_images']

        # extra: parse base folder for images to look out for
        # self.__parse_base_folder()        #TODO: not done in constructor, since an unnecessary model instance is loaded on the AIController side at the moment...

    
    def __parse_base_folder(self):

        all_images = []

        # retrieve all images
        generators = [
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.JPG'), recursive=True),
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.jpg'), recursive=True),
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.JPEG'), recursive=True),
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.jpeg'), recursive=True),
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.PNG'), recursive=True),
            glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.png'), recursive=True)
        ]
        if self.loadRaw:
            generators.append(glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.NEF'), recursive=True))
            generators.append(glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.nef'), recursive=True))
            generators.append(glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.CR2'), recursive=True))
            generators.append(glob.iglob(os.path.join(self.baseFolder_unlabeled, '**/*.cr2'), recursive=True))

        for gen in generators:
            try:
                while True:
                    imgPath = next(gen)
                    all_images.append(imgPath)
            except:
                # end of generator
                pass
        return all_images
    

    def train(self, stateDict, data):
        '''
            TODO: ugly hack: since we do not yet have a model that can cope with all label classes,
            we simply ignore labels the model does not know.
        '''
        if stateDict is not None:
            stateDict_parsed = torch.load(io.BytesIO(stateDict), map_location=lambda storage, loc: storage)
            known_classes = stateDict_parsed['labelclassMap']
        
        self.options['train']['ignore_unsure'] = True
        for key in data['images']:
            nextMeta = data['images'][key]
            if 'annotations' in nextMeta:
                for idx in range(len(nextMeta['annotations'])):
                    anno = data['images'][key]['annotations'][idx]
                    if anno['label'] not in known_classes:
                        data['images'][key]['annotations'][idx]['unsure'] = True
        
        # now start ordinary training
        return super().train(stateDict, data)


    def _inference_image(self, model, transform, filename):
        '''
            Loads the image with given filename from disk, splits it up into
            regular patches, performs inference and then re-splits the image
            into patches that fit the predicted boxes tightly
            ('WindowCropping' strategy), if there are any boxes.
            Commits the resulting patch names to the database and returns the
            identified bounding boxes under the patch names as a dict.
        '''

        # load image
        filePath = os.path.join(self.baseFolder_unlabeled, filename)
        _, fileExt = os.path.splitext(filePath)
        if fileExt.lower() in ['.nef', '.cr2']:
            img = Image.fromarray(rawpy.imread(filePath).postprocess())
        else:
            img = Image.open(filePath).convert('RGB')

        # transform
        tensor = transform(img).to(self._get_device())

        # evaluate in a grid fashion
        gridX, gridY = tensorSharding.createSplitLocations_auto(img.size, [self.patchSize[1], self.patchSize[0]], stride=self.stride, tight=True)
        tensors = tensorSharding.splitTensor(tensor, [self.patchSize[1], self.patchSize[0]], gridY, gridX)
        gridX, gridY = gridX.view(-1).float(), gridY.view(-1).float()

        bboxes = torch.empty(size=(0,4,), dtype=torch.float32)
        labels = torch.empty(size=(0,), dtype=torch.long)
        confs = torch.empty(size=(0, model.numClasses,), dtype=torch.float32)
        scores = torch.empty(size=(0,), dtype=torch.float32)

        numPatches = tensors.size(0)
        numBatches = int(np.ceil(numPatches / float(self.batchSize)))
        for t in range(numBatches):
            startIdx = t*self.batchSize
            endIdx = min((t+1)*self.batchSize, numPatches)
            
            batch = tensors[startIdx:endIdx,:,:,:]
            
            if len(batch.size())==3:
                batch = batch.unsqueeze(0)

            with torch.no_grad():
                bboxes_pred_img, labels_pred_img = model(batch)
            
            bboxes_pred_img, labels_pred_img, confs_pred_img = self.encoder.decode(bboxes_pred_img.squeeze(0).cpu(),
                                                labels_pred_img.squeeze(0).cpu(),
                                                self.patchSize,
                                                cls_thresh=0.1, nms_thresh=0,    #TODO
                                                return_conf=True)

            # incorporate patch offsets and append to list of predictions
            for b in range(len(bboxes_pred_img)):
                if len(bboxes_pred_img[b]):
                    bboxes_pred_img[b][:,0] += gridX[startIdx+b]
                    bboxes_pred_img[b][:,1] += gridY[startIdx+b]
                    bboxes_pred_img[b][:,2] += gridX[startIdx+b]
                    bboxes_pred_img[b][:,3] += gridY[startIdx+b]

                    scores_pred, _ = torch.max(confs_pred_img[b],1)

                    bboxes = torch.cat((bboxes, bboxes_pred_img[b]), dim=0)
                    labels = torch.cat((labels, labels_pred_img[b]), dim=0)
                    confs = torch.cat((confs, confs_pred_img[b]), dim=0)
                    scores = torch.cat((scores, scores_pred), dim=0)

        # do NMS on entire set
        keep = box_nms(bboxes, scores, threshold=0.1)   #TODO
        bboxes = bboxes[keep,:]
        labels = labels[keep]
        confs = confs[keep,:]
        scores = scores[keep]

        # re-split into patches (WindowCropping)
        patchData = self.windowCropper.splitImageIntoPatches(img, bboxes, labels, confs)

        # #TODO
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Rectangle
        # plt.figure(1)
        # plt.clf()
        # plt.imshow(img)
        # ax = plt.gca()
        # for b in range(bboxes.size(0)):
        #     ax.add_patch(Rectangle(
        #         (bboxes[b,0], bboxes[b,1]),
        #         (bboxes[b,2] - bboxes[b,0]), (bboxes[b,3] - bboxes[b,1]),
        #         fill=False,
        #         ec='r'
        #     ))
        # plt.draw()
        # plt.waitforbuttonpress()

        # iterate over patches
        result = {}
        for key in patchData.keys():

            if not self.options['contrib']['export_empty_patches'] and not len(patchData[key]['predictions']):
                continue

            # patch name
            patchName = re.sub('\..*$', '', filename) + '_' + key + '.JPG'
            
            patchDir = os.path.join(self.config.getProperty('FileServer', 'staticfiles_dir'), patchName)
            parentFolder, _ = os.path.split(patchDir)
            os.makedirs(parentFolder, exist_ok=True)

            # save patch
            patchData[key]['patch'].save(patchDir)

            # append metadata
            result[patchName] = {
                'predictions': patchData[key]['predictions']
            }

            # #TODO
            # plt.figure(2)
            # plt.clf()
            # plt.imshow(patchData[key]['patch'])
            # psz = patchData[key]['patch'].size
            # ax = plt.gca()
            # for b in range(len(patchData[key]['predictions'])):
            #     bbox = patchData[key]['predictions'][b]
            #     ax.add_patch(Rectangle(
            #         (psz[0] * (bbox['x']-bbox['width']/2), psz[1] * (bbox['y']-bbox['height']/2),),
            #         psz[0]*bbox['width'], psz[0]*bbox['height'],
            #         fill=False,
            #         ec='r'
            #     ))
            # plt.draw()
            # plt.waitforbuttonpress()

        # return metadata
        return result


    def inference(self, stateDict, data):
        '''
            Augmented implementation of RetinaNet's regular inference function.
            In addition to (or instead of, depending on the settings) performing
            inference on images already existing in the database, the model runs
            over large images specified in the folder ('all_images') and adds the
            predicted patches to the database.
            TODO: Requires to be running on the same instance as the FileServer.
        '''

        # prepare return metadata
        print('Doing inference on new images...')
        response = {}

        # initialize model
        if stateDict is None:
            raise Exception('No trained model state found, but required for inference.')

        # read state dict from bytes
        stateDict_parsed = io.BytesIO(stateDict)
        stateDict_parsed = torch.load(stateDict_parsed, map_location=lambda storage, loc: storage)
        model = Model.loadFromStateDict(stateDict_parsed)
        model.to(self._get_device())

        # mapping labelClass (UUID) to index in model (number); create inverse
        labelclassMap = stateDict_parsed['labelclassMap']
        labelclassMap_inv = {}
        for key in labelclassMap.keys():
            val = labelclassMap[key]
            labelclassMap_inv[val] = key

        # get all image filenames from DB
        current_task.update_state(state='PREPARING', meta={'message':'identifying images'})
        sql = 'SELECT filename FROM {schema}.image;'.format(schema=self.config.getProperty('Database', 'schema'))
        filenames = self.dbConnector.execute(sql, None, 'all')
        filenames = [f['filename'] for f in filenames]

        # get valid filename substring (pattern: path/base_x_y_w_h.JPG)
        fileSnippets_db = set([re.sub('_[0-9]+_[0-9]+_[0-9]+_[0-9]+\..*$', '', f) for f in filenames])

        # the same for images on disk
        images_disk = self.__parse_base_folder()
        snippets_disk = [os.path.splitext(f.replace(self.baseFolder_unlabeled, '')) for f in images_disk]
        extensions_disk = dict(snippets_disk)
        fileSnippets_disk = set(extensions_disk.keys())

        # identify images that have not yet been added to DB
        unlabeled = fileSnippets_disk.difference(fileSnippets_db)

        if not len(unlabeled):
            return response

        # choose n unlabeled
        unlabeled = random.sample(unlabeled, min(self.maxNumUnlabeled, len(unlabeled)))

        # prepare transforms
        transform = tr.Compose([
            tr.ToTensor(),
            tr.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])  #TODO: write functional.pytorch util to compose transformations

        # do inference on the unlabeled images
        for u in tqdm(range(len(unlabeled))):
            #TODO
            print(unlabeled[u] + extensions_disk[unlabeled[u]])
            meta = self._inference_image(model, transform, unlabeled[u] + extensions_disk[unlabeled[u]])

            for key in meta.keys():
                response[key] = meta[key]

            # update worker state
            current_task.update_state(state='PROGRESS', meta={'done': u+1, 'total': len(unlabeled), 'message': 'predicting new images'})
    
        model.cpu()
        if 'cuda' in self._get_device():
            torch.cuda.empty_cache()


        if len(response.keys()):
            # convert label indices to UUIDs
            for key in response.keys():
                for p in range(len(response[key]['predictions'])):
                    response[key]['predictions'][p]['label'] = labelclassMap_inv[response[key]['predictions'][p]['label']]

            # commit to DB
            current_task.update_state(state='PREPARING', meta={'message':'adding new images to database'})
            sql = '''
                INSERT INTO {schema}.image (filename)
                VALUES %s;
            '''.format(schema=self.config.getProperty('Database', 'schema'))
            self.dbConnector.insert(sql, [(key,) for key in list(response.keys())])

            
            # get IDs of newly inserted patches
            sql = '''
                SELECT id, filename FROM {schema}.image WHERE filename IN %s;
            '''.format(schema=self.config.getProperty('Database', 'schema'))
            patchIDs = self.dbConnector.execute(sql, (tuple(response.keys()),), 'all')

            # replace IDs of responses
            new_response = {}
            for p in patchIDs:
                new_response[p['id']] = response[p['filename']]
            response = new_response


        # also do regular inference
        print('Doing inference on existing patches...')
        response_regular = super(RetinaNet_ois, self).inference(stateDict, data)
        for key in response_regular.keys():
            response[key] = response_regular[key]

        return response



# #TODO
# if __name__ == '__main__':


#     os.environ['AIDE_CONFIG_PATH'] = 'settings_windowCropping.ini'
#     from util.configDef import Config
#     from modules.Database.app import Database
#     from modules.AIWorker.backend.worker.fileserver import FileServer
#     config = Config()
#     dbConnector = Database(config)
#     fileServer = FileServer(config)

#     rn = RetinaNet_ois(config, dbConnector, fileServer, None)


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

#     sql = '''SELECT image FROM aerialelephants_wc.image_user WHERE viewcount > 0 LIMIT 2''' 
#     imageIDs = dbConnector.execute(sql, None, 2)
#     imageIDs = [i['image'] for i in imageIDs]

#     data = __load_metadata(config, dbConnector, imageIDs, False)

#     # stateDict = rn.train(stateDict, data)

#     print('debug')

#     result = rn.inference(stateDict, data)