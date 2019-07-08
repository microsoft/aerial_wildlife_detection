#TODO: plug in RetinaNet implementation eventually...

#TODO 2: define function shells first in an abstract superclass (with documentation) and a template.

from celery import current_task
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms as tr
from ai.functional.pytorch._retinanet import collation, encoder, transforms as bboxTr, loss
from ai.functional.pytorch._retinanet.model import RetinaNet as Model
from ai.functional.pytorch.datasets.bboxDataset import BoundingBoxDataset


class RetinaNet:

    def __init__(self, config, dbConnector, fileServer, options):
        #TODO
        print('calling retinanet constructor')
        self.config = config
        self.dbConnector = dbConnector
        self.fileServer = fileServer
        self.options = options
    

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
            model = Model.loadFromStateDict(self.options)   #TODO: also provide default options / load from properties file
        

        # initialize data loader, dataset, transforms, optimizer, criterion
        transforms = bboxTr.Compose([
            bboxTr.RandomHorizontalFlip(p=0.5),
            bboxTr.DefaultTransform(tr.ColorJitter(0.25, 0.25, 0.25, 0.01)),
            bboxTr.DefaultTransform(tr.ToTensor()),
            bboxTr.DefaultTransform(tr.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]))
        ])  #TODO: ditto, also write functional.pytorch util to compose transformations
        dataset = BoundingBoxDataset(data,
                                    stateDict,
                                    self.fileServer,
                                    targetFormat='xywh',
                                    transform=transforms,
                                    ignoreUnsure=self.options['ignore_unsure'])  #TODO: ditto
        dataEncoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   #TODO: ditto
        inputSize = (512, 512,)     #TODO: ditto
        collator = collation.Collator(inputSize, dataEncoder)
        dataLoader = DataLoader(
            dataset,
            batch_size=self.options['batch_size'],   #TODO: ditto
            shuffle=True,
            collate_fn=collator.collate_fn
        )
        optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=0.0)       #TODO: ditto (write functional.pytorch util to load optim)
        criterion = loss.FocalLoss(num_classes=len(dataset.classdef), alpha=0.25, gamma=2, classWeights=None)     #TODO: ditto


        # train model
        device = 'cuda:0'
        #TODO: outsource into dedicated function; set GPU, set random seed, etc.
        for idx, (img, _, bboxes_target, labels_target) in enumerate(dataLoader):
            img, boundingBoxes, labels = img.to(device), boundingBoxes.to(device), labels.to(device)

            optimizer.zero_grad()
            bboxes_pred, labels_pred = model(img)
            loss_value = criterion(bboxes_pred, bboxes_target, labels_pred, labels_target)
            loss_value.backward()
            optimizer.step()

            # update worker state   another TODO
            current_task.update_state(state='PROGRESS', meta={'done': idx+1, 'total': len(dataLoader)})


        # all done; return state dict
        stateDict = model.getStateDict()

        return stateDict


    def average_model_states(self, stateDicts):
        '''
            TODO
        '''
        average_states = Model.averageStateDicts(stateDicts)
        return average_states

    
    def inference(self, stateDict, data):
        '''
            TODO
        '''

        # initialize model
        model = Model.loadFromStateDict(stateDict)

        # initialize data loader, dataset, transforms
        transforms = bboxTr.Compose([
            bboxTr.DefaultTransform(tr.ToTensor()),
            bboxTr.DefaultTransform(tr.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]))
        ])  #TODO: ditto, also write functional.pytorch util to compose transformations
        dataset = BoundingBoxDataset(data,
                                    stateDict,
                                    self.fileServer,
                                    targetFormat='xywh',
                                    transform=transforms)  #TODO: ditto
        dataEncoder = encoder.DataEncoder(minIoU_pos=0.5, maxIoU_neg=0.4)   #TODO: ditto
        inputSize = (512, 512,)     #TODO: ditto
        collator = collation.Collator(inputSize, dataEncoder)
        dataLoader = DataLoader(
            dataset,
            batch_size=self.options['batch_size'],   #TODO: ditto
            shuffle=True,
            collate_fn=collator.collate_fn
        )

        # perform inference
        response = {}
        device = 'cuda:0'
        #TODO: outsource into dedicated function; set GPU, set random seed, etc.
        for idx, (img, imgID, _, _) in enumerate(dataLoader):
            img = img.to(device)

            with torch.no_grad():
                bboxes_pred, labels_pred = model(img)
                bboxes_pred_img, labels_pred_img, confs_pred_img = dataEncoder.decode(bboxes_pred.squeeze(0).cpu(),
                                    labels_pred.squeeze(0).cpu(),
                                    (inputSize[1],inputSize[0],),
                                    cls_thresh=0.1, nms_thresh=0.1,
                                    return_conf=True)       #TODO: ditto

                # append to dict
                response[imgID] = {
                    'predictions': {}       #TODO
                }

            # update worker state   another TODO
            current_task.update_state(state='PROGRESS', meta={'done': idx+1, 'total': len(dataLoader)})

        return response