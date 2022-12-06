'''
    2021 Benjamin Kellenberger
'''

from psycopg2 import sql
from tqdm import tqdm
import torch
from detectron2.data import build_detection_test_loader

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2._functional.datasetMapper import Detectron2DatasetMapper
from ai.models.detectron2._functional.dataset import getDetectron2Data


class GenericDetectron2LabelModel(GenericDetectron2Model):

    def calculateClassCorrelations(self, stateDict, model, labelclassMap, modelClasses, targetClasses, updateStateFun, maxNumImages=None):
        '''
            Class correlation implementation for image label prediction models.
        '''
        #TODO: safety checks; update state fun; etc.
        

        # query data
        queryArgs = [tuple((l,) for l in targetClasses)]
        if isinstance(maxNumImages, int):
            limitStr = sql.SQL('LIMIT %s')
            queryArgs.append(maxNumImages)
        else:
            limitStr = sql.SQL('')

        queryStr = sql.SQL('''
            SELECT image, filename, label
            FROM {id_anno} AS a
            JOIN {id_img} AS img
            ON a.image = img.id
            WHERE label IN %s
            {limitStr}
        ''').format(
            id_anno=sql.Identifier(self.project, 'annotation'),
            id_img=sql.Identifier(self.project, 'image'),
            limitStr=limitStr
        )
        result = self.dbConnector.execute(queryStr, tuple(queryArgs), 'all')
        imgDict = {}
        for r in result:
            imgID = r['image']
            if imgID not in imgDict:
                imgDict[imgID] = {
                    'filename': r['filename'],
                    'annotations': [r]
                }
        data = {
            'labelClasses': targetClasses,
            'images': imgDict
        }

        # prepare result vector
        correlations = torch.zeros(len(targetClasses), len(modelClasses))
        counts = correlations.clone()

        # prepare data loader
        transforms = self.initializeTransforms(mode='inference')
        
        bandConfig = self._get_band_config(stateDict, data)
        # try:
        #     imageFormat = self.detectron2cfg.INPUT.FORMAT
        #     assert imageFormat.upper() in ('RGB', 'BGR')
        # except Exception:
        #     imageFormat = 'BGR'
        datasetMapper = Detectron2DatasetMapper(self.project, self.fileServer, transforms, False, bandConfig)
        dataLoader = build_detection_test_loader(
            dataset=getDetectron2Data(data, labelclassMap, {}, False, False),
            mapper=datasetMapper,
            num_workers=0
        )
        dataLoaderIter = iter(dataLoader)
        numImgs = len(imgDict)

        # perform forward pass over all images
        updateStateFun(state='PROGRESS', message='predicting images with new label classes')
        model.eval()
        with torch.no_grad():
            for idx in tqdm(range(numImgs)):
                try:
                    batch = next(dataLoaderIter)
                except Exception:
                    break

                outputs = model(batch)
                pred_logits = outputs[0]['pred_logits']
                pred_label = outputs[0]['pred_label']

                # calc. correlation per target instance and label class
                gt_label = batch[0]['gt_label']

                correlations[gt_label,:] += pred_logits.cpu()
                counts[gt_label,pred_label] += 1

                updateStateFun(state='PROGRESS', message='predicting', done=(idx+1), total=numImgs)
            
            # average correlations
            counts = counts / counts.sum(1, keepdim=True)
            correlations *= counts

            # normalize
            correlations /= correlations.sum(1, keepdim=True)
            
            return correlations