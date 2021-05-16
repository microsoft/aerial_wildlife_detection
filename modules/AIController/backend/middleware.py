'''
    Middleware for AIController: handles requests and updates to and from the database.

    2019-21 Benjamin Kellenberger
'''

from collections.abc import Iterable
from datetime import datetime
import uuid
import re
import json
from constants.annotationTypes import ANNOTATION_TYPES
from ai import PREDICTION_MODELS, ALCRITERION_MODELS
from modules.AIController.backend import celery_interface as aic_int
from modules.AIWorker.backend import celery_interface as aiw_int
from celery import current_app
from psycopg2 import sql
from .messageProcessor import MessageProcessor
from .annotationWatchdog import Watchdog
from modules.AIController.taskWorkflow.workflowDesigner import WorkflowDesigner
from modules.AIController.taskWorkflow.workflowTracker import WorkflowTracker
from modules.AIWorker.backend.fileserver import FileServer
from util import celeryWorkerCommons
from util.helpers import array_split, parse_parameters, get_class_executable, get_library_available

from .sql_string_builder import SQLStringBuilder


class AIMiddleware():

    def __init__(self, config, dbConnector, taskCoordinator, passiveMode=False):
        self.config = config
        self.dbConn = dbConnector
        self.taskCoordinator = taskCoordinator
        self.sqlBuilder = SQLStringBuilder(config)
        self.passiveMode = passiveMode
        self.scriptPattern = re.compile(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script\.?>')
        self._init_available_ai_models()

        self.celery_app = current_app
        self.celery_app.set_current()
        self.celery_app.set_default()
        
        if not self.passiveMode:
            self.messageProcessor = MessageProcessor(self.celery_app)
            self.watchdogs = {}    # one watchdog per project. Note: watchdog only created if users poll status (i.e., if there's activity)
            self.workflowDesigner = WorkflowDesigner(self.dbConn, self.celery_app)
            self.workflowTracker = WorkflowTracker(self.dbConn, self.celery_app)
            self.messageProcessor.start()

    
    def __del__(self):
        if self.passiveMode:
            return
        self.messageProcessor.stop()
        for w in self.watchdogs.values():
            w.stop()


    def _init_available_ai_models(self):
        # for built-in models: check if Detectron2 and PyTorch are installed
        hasPyTorch = get_library_available('torch')
        hasDetectron = get_library_available('detectron2')

        #TODO: 1. using regex to remove scripts is not failsafe; 2. ugly code...
        models = {
            'prediction': PREDICTION_MODELS,
            'ranking': ALCRITERION_MODELS
        }

        # remove script tags and validate annotation and prediction types specified
        for modelKey in models['prediction']:
            model = models['prediction'][modelKey]
            if 'name' in model and isinstance(model['name'], str):
                model['name'] = re.sub(self.scriptPattern, '(script removed)', model['name'])
            else:
                model['name'] = modelKey
            if 'description' in model and isinstance(model['description'], str):
                model['description'] = re.sub(self.scriptPattern, '(script removed)', model['description'])
            else:
                model['description'] = '(no description available)'
            if 'author' in model and isinstance(model['author'], str):
                model['author'] = re.sub(self.scriptPattern, '(script removed)', model['author'])
            else:
                model['author'] = '(unknown)'
            
            # check required libraries
            if model['author'] == '(built-in)':
                if '.pytorch.' in modelKey.lower() and not hasPyTorch:
                    print(f'WARNING: model "{modelKey}" requires PyTorch library, which is not installed, and is therefore ignored.')
                    del models['prediction'][modelKey]
                    continue
                elif '.detectron2.' in modelKey.lower() and not hasDetectron:
                    print(f'WARNING: model "{modelKey}" requires Detectron2 library, which is not installed, and is therefore ignored.')
                    del models['prediction'][modelKey]
                    continue

            if not 'annotationType' in model or not 'predictionType' in model:
                # no annotation and/or no prediction type specified; remove model
                print(f'WARNING: model "{modelKey}" has no annotationType and/or no predictionType specified and is therefore ignored.')
                del models['prediction'][modelKey]
                continue
            if isinstance(model['annotationType'], str):
                model['annotationType'] = [model['annotationType']]
            for idx, type in enumerate(model['annotationType']):
                if type not in ANNOTATION_TYPES:
                    print(f'WARNING: annotation type "{type}" not understood and ignored.')
                    del model['annotationType'][idx]
            if isinstance(model['predictionType'], str):
                model['predictionType'] = [model['predictionType']]
            for idx, type in enumerate(model['predictionType']):
                if type not in ANNOTATION_TYPES:
                    print(f'WARNING: prediction type "{type}" not understood and ignored.')
                    del model['predictionType'][idx]
            if model['annotationType'] is None or not len(model['annotationType']):
                print(f'WARNING: no valid annotation type specified for model "{modelKey}"; ignoring...')
                del models['prediction'][modelKey]
                continue
            if model['predictionType'] is None or not len(model['predictionType']):
                print(f'WARNING: no valid prediction type specified for model "{modelKey}"; ignoring...')
                del models['prediction'][modelKey]
                continue
            # default model options
            try:
                modelClass = get_class_executable(modelKey)
                defaultOptions = modelClass.getDefaultOptions()
                model['defaultOptions'] = defaultOptions
            except:
                # no default options available; append no key to signal that there's no options
                pass
            models['prediction'][modelKey] = model
        for rankerKey in models['ranking']:
            ranker = models['ranking'][rankerKey]
            if 'name' in ranker and isinstance(ranker['name'], str):
                ranker['name'] = re.sub(self.scriptPattern, '(script removed)', ranker['name'])
            else:
                ranker['name'] = rankerKey
            if 'author' in ranker and isinstance(ranker['author'], str):
                ranker['author'] = re.sub(self.scriptPattern, '(script removed)', ranker['author'])
            else:
                ranker['author'] = '(unknown)'
            if 'description' in ranker and isinstance(ranker['description'], str):
                ranker['description'] = re.sub(self.scriptPattern, '(script removed)', ranker['description'])
            else:
                ranker['description'] = '(no description available)'
            if not 'predictionType' in ranker:
                # no prediction type specified; remove ranker
                print(f'WARNING: ranker "{rankerKey}" has no predictionType specified and is therefore ignored.')
                del models['ranking'][rankerKey]
                continue
            if isinstance(ranker['predictionType'], str):
                ranker['predictionType'] = [ranker['predictionType']]
            for idx, type in enumerate(ranker['predictionType']):
                if type not in ANNOTATION_TYPES:
                    print(f'WARNING: prediction type "{type}" not understood and ignored.')
                    del ranker['predictionType'][idx]
            if ranker['predictionType'] is None or not len(ranker['predictionType']):
                print(f'WARNING: no valid prediction type specified for ranker "{rankerKey}"; ignoring...')
                del models['ranking'][rankerKey]
                continue
            # default ranker options
            try:
                rankerClass = get_class_executable(rankerKey)
                defaultOptions = rankerClass.getDefaultOptions()
                ranker['defaultOptions'] = defaultOptions
            except:
                # no default options available; append no key to signal that there's no options
                pass
            models['ranking'][rankerKey] = ranker
        self.aiModels = models



    def _init_watchdog(self, project, nudge=False, recheckAutotrainSettings=False):
        '''
            Launches a thread that periodically polls the database for new
            annotations. Once the required number of new annotations is reached,
            this thread will initiate the training process through the middleware.
            The thread will be terminated and destroyed; a new thread will only be
            re-created once the training process has finished.
        '''
        if self.passiveMode:
            return

        if project not in self.watchdogs:
            self.watchdogs[project] = Watchdog(project, self.config, self.dbConn, self)
            self.watchdogs[project].start()   
        
        if recheckAutotrainSettings:
            # also nudges the watchdog
            self.watchdogs[project].recheckAutotrainSettings()
        
        elif nudge:
            self.watchdogs[project].nudge()



    def _get_num_available_workers(self):
        #TODO: message + queue if no worker available
        #TODO: limit to n tasks per worker
        i = self.celery_app.control.inspect()
        if i is not None:
            stats = i.stats()
            if stats is not None:
                return len(i.stats())
        return 1    #TODO



    def _get_project_settings(self, project):
        queryStr = sql.SQL('''SELECT numImages_autoTrain,
            minNumAnnoPerImage, maxNumImages_train,maxNumImages_inference
            FROM aide_admin.project WHERE shortname = %s;''')
        settings = self.dbConn.execute(queryStr, (project,), 1)[0]
        return settings



    def get_ai_model_training_info(self, project):
        '''
            Returns information required to determine whether AI models can be trained
            for a given project.
            This includes:
                - Whether an AI model library is configured for the project
                - Whether at least consumer for each AIController and AIWorker is
                  connected and available
            Returns a dict of this information accordingly.
        '''
        # check whether project has an AI model configured
        aiModelLibrary = self.dbConn.execute('''
            SELECT ai_model_library
            FROM aide_admin.project
            WHERE shortname = %s;
        ''', (project,), 1)
        try:
            aiModelLibrary = aiModelLibrary[0]['ai_model_library']
        except:
            aiModelLibrary = None
        
        # check if AIController worker and AIWorker are connected
        aicW = {}
        aiwW = {}
        workers = celeryWorkerCommons.getCeleryWorkerDetails()
        for wk in workers.keys():
            try:
                worker = workers[wk]
                if 'AIController' in worker['modules'] and worker['modules']['AIController'] == True:
                    aicW[wk] = workers[wk]
                if 'AIWorker' in worker['modules'] and worker['modules']['AIWorker'] == True:
                    aiwW[wk] = workers[wk]
            except:
                pass
        
        return {
            'ai_model_library': aiModelLibrary,
            'workers': {
                'AIController': aicW,
                'AIWorker': aiwW
            }
        }



    def get_ongoing_tasks(self, project):
        '''
            Polls Celery via Watchdog and returns a list of IDs of tasks
            that are currently ongoing for the respective project.
        '''
        self._init_watchdog(project)
        return self.watchdogs[project].getOngoingTasks()

    
    
    def can_launch_task(self, project, autoLaunched):
        '''
            Polls ongoing tasks for the project in question and retrieves
            the maximum number of tasks that are allowed to be executed
            concurrently (as per project settings). Returns True if one
            (more) task can be launched, and False otherwise.
            Only one auto-launched taks can be run at a time ("autoLaunched"
            True). The number of user-launched tasks depends on the project
            settings.
        '''
        # query number of currently ongoing tasks
        ongoingTasks = self.get_ongoing_tasks(project)
        if autoLaunched and len(ongoingTasks):
            # we only permit one auto-launched task at a time
            return False
        
        # query number of concurrent tasks allowed as per project settings
        upperCeiling = self.config.getProperty('AIController', 'max_num_concurrent_tasks', type=int, fallback=2)
        numConcurrent = self.dbConn.execute('''
            SELECT max_num_concurrent_tasks
            FROM aide_admin.project
            WHERE shortname = %s;
        ''', (project,), 1)
        try:
            numConcurrent = numConcurrent[0]['max_num_concurrent_tasks']
            if upperCeiling > 0:
                numConcurrent = min(numConcurrent, upperCeiling)
        except:
            numConcurrent = upperCeiling
        
        if numConcurrent <= 0:
            return True
        else:
            return len(ongoingTasks) < numConcurrent



    def aide_internal_notify(self, message):
        '''
            Used for AIDE administrative communication between AIController
            and AIWorker(s), e.g. for setting up queues.
        '''
        if self.passiveMode:
            return
        #TODO: not required (yet)



    def get_training_images(self, project, minTimestamp='lastState', includeGoldenQuestions=True,
                            minNumAnnoPerImage=0, maxNumImages=None, maxNumWorkers=-1):
        '''
            Queries the database for the latest images to be used for model training.
            Returns a list with image UUIDs accordingly, split into the number of
            available workers.
        '''
        # sanity checks
        if not (isinstance(minTimestamp, datetime) or minTimestamp == 'lastState' or
                minTimestamp == -1 or minTimestamp is None):
            raise ValueError('{} is not a recognized property for variable "minTimestamp"'.format(str(minTimestamp)))

        # identify number of available workers
        if maxNumWorkers != 1:
            # only query the number of available workers if more than one is specified to save time
            num_workers = min(maxNumWorkers, self._get_num_available_workers())
        else:
            num_workers = maxNumWorkers

        # query image IDs
        queryVals = []

        if minTimestamp is None:
            timestampStr = sql.SQL('')
        elif minTimestamp == 'lastState':
            timestampStr = sql.SQL('''
            WHERE iu.last_checked > COALESCE(to_timestamp(0),
            (SELECT MAX(timecreated) FROM {id_cnnstate}))''').format(
                id_cnnstate=sql.Identifier(project, 'cnnstate')
            )
        elif isinstance(minTimestamp, datetime):
            timestampStr = sql.SQL('WHERE iu.last_checked > COALESCE(to_timestamp(0), %s)')
            queryVals.append(minTimestamp)
        elif isinstance(minTimestamp, int) or isinstance(minTimestamp, float):
            timestampStr = sql.SQL('WHERE iu.last_checked > COALESCE(to_timestamp(0), to_timestamp(%s))')
            queryVals.append(minTimestamp)
        
        # golden questions
        if includeGoldenQuestions:
            gqStr = sql.SQL('')
        else:
            gqStr = sql.SQL('AND isGoldenQuestion != TRUE')

        if minNumAnnoPerImage > 0:
            queryVals.append(minNumAnnoPerImage)

        if maxNumImages is None or not isinstance(maxNumImages, int) or maxNumImages <= 0:
            limitStr = sql.SQL('')
        else:
            limitStr = sql.SQL('LIMIT %s')
            queryVals.append(maxNumImages)

        if minNumAnnoPerImage <= 0:
            queryStr = sql.SQL('''
                SELECT newestAnno.image FROM (
                    SELECT image, last_checked FROM {id_iu} AS iu
                    JOIN (
                        SELECT id AS iid
                        FROM {id_img}
                        WHERE (corrupt IS NULL OR corrupt = FALSE)
                        {gqStr}
                    ) AS imgQ
                    ON iu.image = imgQ.iid
                    {timestampStr}
                    ORDER BY iu.last_checked ASC
                    {limitStr}
                ) AS newestAnno;
            ''').format(
                id_iu=sql.Identifier(project, 'image_user'),
                id_img=sql.Identifier(project, 'image'),
                gqStr=gqStr,
                timestampStr=timestampStr,
                limitStr=limitStr)

        else:
            queryStr = sql.SQL('''
                SELECT newestAnno.image FROM (
                    SELECT image, last_checked FROM {id_iu} AS iu
                    JOIN (
                        SELECT id AS iid
                        FROM {id_img}
                        WHERE corrupt IS NULL OR corrupt = FALSE
                    ) AS imgQ
                    ON iu.image = imgQ.iid
                    {timestampStr}
                    {conjunction} image IN (
                        SELECT image FROM (
                            SELECT image, COUNT(*) AS cnt
                            FROM {id_anno}
                            GROUP BY image
                            ) AS annoCount
                        WHERE annoCount.cnt >= %s
                    )
                    ORDER BY iu.last_checked ASC
                    {limitStr}
                ) AS newestAnno;
            ''').format(
                id_iu=sql.Identifier(project, 'image_user'),
                id_img=sql.Identifier(project, 'image'),
                id_anno=sql.Identifier(project, 'annotation'),
                timestampStr=timestampStr,
                conjunction=(sql.SQL('WHERE') if minTimestamp is None else sql.SQL('AND')),
                limitStr=limitStr)

        imageIDs = self.dbConn.execute(queryStr, tuple(queryVals), 'all')
        imageIDs = [i['image'] for i in imageIDs]

        if maxNumWorkers > 1:
            # split for distribution across workers (TODO: also specify subset size for multiple jobs; randomly draw if needed)
            imageIDs = array_split(imageIDs, max(1, len(imageIDs) // num_workers))
        else:
            imageIDs = [imageIDs]

        return imageIDs



    def get_inference_images(self, project, goldenQuestionsOnly=False, forceUnlabeled=False, maxNumImages=None, maxNumWorkers=-1):
        '''
            Queries the database for the latest images to be used for inference after model training.
            Returns a list with image UUIDs accordingly, split into the number of available workers.
        '''
        if maxNumImages is None or maxNumImages == -1:
            queryResult = self.dbConn.execute('''
                SELECT maxNumImages_inference
                FROM aide_admin.project
                WHERE shortname = %s;''', (project,), 1)
            maxNumImages = queryResult[0]['maxnumimages_inference']    
        
        queryVals = (maxNumImages,)

        # load the IDs of the images that are being subjected to inference
        sql = self.sqlBuilder.getInferenceQueryString(project, forceUnlabeled, goldenQuestionsOnly, maxNumImages)
        imageIDs = self.dbConn.execute(sql, queryVals, 'all')
        imageIDs = [i['image'] for i in imageIDs]

        # split for distribution across workers
        if maxNumWorkers != 1:
            # only query the number of available workers if more than one is specified to save time
            num_available = self._get_num_available_workers()
            if maxNumWorkers == -1:
                maxNumWorkers = num_available   #TODO: more than one process per worker?
            else:
                maxNumWorkers = min(maxNumWorkers, num_available)
        
        if maxNumWorkers > 1:
            imageIDs = array_split(imageIDs, max(1, len(imageIDs) // maxNumWorkers))
        else:
            imageIDs = [imageIDs]
        return imageIDs
    


    #TODO: deprecated; replace with workflow
    def start_train_and_inference(self, project, minTimestamp='lastState', minNumAnnoPerImage=0, maxNumImages_train=-1, 
                                    maxNumWorkers_train=1,
                                    forceUnlabeled_inference=True, maxNumImages_inference=-1, maxNumWorkers_inference=1,
                                    author=None):
        '''
            Submits a model training job, followed by inference.
            This is the default behavior for the automated model update, since the newly trained model should directly
            be used to infer new, potentially useful labels.
        '''
        # check if task launching is allowed
        if not self.can_launch_task(project, None):
            return {
                'status': 1,
                'message': f'The maximum allowed number of concurrent tasks has been reached for project "{project}". Please wait until running tasks are finished.'
            }

        #TODO: sanity checks
        workflow = {
            'project': project,
            'tasks': [
                {
                    'id': '0',
                    'type': 'train',
                    'kwargs': {
                        'min_timestamp': minTimestamp,
                        'min_anno_per_image': minNumAnnoPerImage,
                        'max_num_images': maxNumImages_train,
                        'max_num_workers': maxNumWorkers_train
                    }
                },
                {
                    'id': '1',
                    'type': 'inference',
                    'kwargs': {
                        'force_unlabeled': forceUnlabeled_inference,
                        'max_num_images': maxNumImages_inference,
                        'max_num_workers': maxNumWorkers_inference
                    }
                }
            ],
            'options': {}
        }
        process = self.workflowDesigner.parseWorkflow(project, workflow)

        
        # launch workflow
        task_id = self.workflowTracker.launchWorkflow(project, process, workflow, author)

        return {
            'status': 0,
            'task_id': task_id
        }


    
    def launch_task(self, project, workflow, author=None):
        '''
            Accepts a workflow as one of the following three variants:
            - ID (str or UUID): ID of a saved workflow in this project
            - 'default': uses the default workflow for this project
            - dict: an actual workflow as per specifications
            parses it and launches the job if valid.
            Returns the task ID accordingly.
        '''
        # check if task launching is allowed
        if not self.can_launch_task(project, author):
            return {
                'status': 1,
                'message': f'The maximum allowed number of concurrent tasks has been reached for project "{project}". Please wait until running tasks are finished.'
            }

        if isinstance(workflow, str):
            if workflow.lower() == 'default':
                # load default workflow
                queryStr = sql.SQL('''
                    SELECT workflow FROM {id_workflow}
                    WHERE id = (
                        SELECT default_workflow
                        FROM aide_admin.project
                        WHERE shortname = %s
                    );
                ''').format(
                    id_workflow=sql.Identifier(project, 'workflow')
                )
                result = self.dbConn.execute(queryStr, (project,), 1)
                if result is None or not len(result):
                    return {
                        'status': 2,
                        'message': f'Workflow with ID "{str(workflow)}" does not exist in this project'
                    }
                workflow = result[0]['workflow']
            else:
                # try first to parse workflow
                try:
                    workflow = json.loads(workflow)
                except:
                    # try to convert to UUID instead
                    try:
                        workflow = uuid.UUID(workflow)
                    except:
                        return {
                            'status': 3,
                            'message': f'"{str(workflow)}" is not a valid workflow ID'
                        }
        
        if isinstance(workflow, uuid.UUID):
            # load workflow as per UUID
            queryStr = sql.SQL('''
                    SELECT workflow FROM {id_workflow}
                    WHERE id = %s;
                ''').format(
                    id_workflow=sql.Identifier(project, 'workflow')
                )
            result = self.dbConn.execute(queryStr, (workflow,), 1)
            if result is None or not len(result):
                return {
                    'status': 2,
                    'message': f'Workflow with ID "{str(workflow)}" does not exist in this project'
                }
            workflow = result[0]['workflow']

        # try to parse workflow
        try:
            process = self.workflowDesigner.parseWorkflow(project, workflow, False)
        except Exception as e:
            return {
                'status': 4,
                'message': f'Workflow could not be parsed (message: "{str(e)}")'
            }

        task_id = self.workflowTracker.launchWorkflow(project, process, workflow, author)

        return {
            'status': 0,
            'task_id': task_id
        }



    def revoke_task(self, project, taskID, username):
        '''
            Revokes (aborts) a task with given task ID for a given
            project, if it exists.
            Also sets an entry in the database (and notes who aborted
            the task).
        '''
        self.workflowTracker.revokeTask(username, project, taskID)

    

    def revoke_all_tasks(self, project, username):
        '''
            Revokes (aborts) all tasks for a given project.
            Also sets an entry in the database (and notes who aborted
            the task).
        '''
        #TODO: make more elegant
        taskIDs = self.get_ongoing_tasks(project)
        for taskID in taskIDs:
            self.revoke_task(project, taskID, username)



    def check_status(self, project, checkProject, checkTasks, checkWorkers, nudgeWatchdog=False, recheckAutotrainSettings=False):
        '''
            Queries the Celery worker results depending on the parameters specified.
            Returns their status accordingly if they exist.
        '''
        status = {}

        # watchdog
        self._init_watchdog(project, nudgeWatchdog, recheckAutotrainSettings)

        # project status
        if checkProject:
            status['project'] = {
                'ai_auto_training_enabled': self.watchdogs[project].getAImodelAutoTrainingEnabled(),
                'num_annotated': self.watchdogs[project].lastCount,
                'num_next_training': self.watchdogs[project].getThreshold()
            }

        # running tasks status
        if checkTasks:
            status['tasks'] = self.workflowTracker.pollAllTaskStatuses(project)       #TODO: self.messageProcessor.poll_status(project)

        # get worker status (this is very expensive, as each worker needs to be pinged)
        if checkWorkers:
            status['workers'] = self.messageProcessor.poll_worker_status()
        
        return status



    def getAvailableAImodels(self, project=None):
        if project is None:
            return {
                'models': self.aiModels
            }
        else:
            models = {
                'prediction': {},
                'ranking': self.aiModels['ranking']
            }

            # identify models that are compatible with project's annotation and prediction type
            projImmutables = self.dbConn.execute(
                '''
                    SELECT annotationtype, predictiontype
                    FROM aide_admin.project
                    WHERE shortname = %s;
                ''',
                (project,),
                1
            )
            projImmutables = projImmutables[0]
            projAnnoType = projImmutables['annotationtype']
            projPredType = projImmutables['predictiontype']
            for key in self.aiModels['prediction'].keys():
                model = self.aiModels['prediction'][key]
                if projAnnoType in model['annotationType'] and \
                    projPredType in model['predictionType']:
                    models['prediction'][key] = model

            return {
                'models': models
            }



    def verifyAImodelOptions(self, project, modelOptions, modelLibrary=None):
        '''
            Receives a dict of model options, a model library ID (optional)
            and verifies whether the provided options are valid for the model
            or not. Uses the following strategies to this end:
            1. If the AI model implements the function "verifyOptions", and
               if that function does not return None, it is being used.
            2. Else, this temporarily instanciates a new AI model with the
               given options and checks whether any errors occur. Returns
               True if not, but appends a warning that the options could not
               reliably be verified.
        '''
        # get AI model library if not specified
        if modelLibrary is None or modelLibrary not in self.aiModels['prediction']:
            modelLib = self.dbConn.execute('''
                SELECT ai_model_library
                FROM aide_admin.project
                WHERE shortname = %s;
            ''', (project,), 1)
            modelLibrary = modelLib[0]['ai_model_library']

        modelName = self.aiModels['prediction'][modelLibrary]['name']

        response = None

        modelClass = get_class_executable(modelLibrary)
        if hasattr(modelClass, 'verifyOptions'):
            response = modelClass.verifyOptions(modelOptions)
        
        if response is None:
            # no verification implemented; alternative
            #TODO: can we always do that on the AIController?
            try:
                modelClass(project=project,
                            config=self.config,
                            dbConnector=self.dbConn,
                            fileServer=FileServer(self.config).get_secure_instance(project),
                            options=modelOptions)
                response = {
                    'valid': True,
                    'warnings': [f'A {modelName} instance could be launched, but the settings could not be verified.']
                }
            except Exception as e:
                # model could not be instantiated; append error
                response = {
                    'valid': False,
                    'errors': [ str(e) ]
                }
        
        return response



    def updateAImodelSettings(self, project, settings):
        '''
            Updates the project's AI model settings.
            Verifies whether the specified AI and ranking model libraries
            exist on this setup of AIDE. Raises an exception otherwise.

            Also tries to verify any model options provided with the
            model's built-in function (if present and implemented).
            Returns warnings, errors, etc. about that.
        '''
        # AI libraries installed in AIDE
        availableModels = self.getAvailableAImodels()['models']

        # project immutables
        projImmutables = self.dbConn.execute(
            '''
                SELECT annotationtype, predictiontype
                FROM aide_admin.project
                WHERE shortname = %s;
            ''',
            (project,),
            1
        )
        projImmutables = projImmutables[0]
        annoType = projImmutables['annotationtype']
        predType = projImmutables['predictiontype']

        # cross-check submitted tokens
        fieldNames = [
            ('ai_model_enabled', bool),
            ('ai_model_library', str),
            ('ai_alcriterion_library', str),
            ('numimages_autotrain', int),           #TODO: replace this and next four entries with default workflow
            ('minnumannoperimage', int),
            ('maxnumimages_train', int),
            ('maxnumimages_inference', int),
            ('inference_chunk_size', int),
            ('max_num_concurrent_tasks', int),
            ('segmentation_ignore_unlabeled', bool)
        ]
        settings_new, settingsKeys_new = parse_parameters(settings, fieldNames, absent_ok=True, escape=True, none_ok=True)

        # verify settings
        addBackgroundClass = False
        forceDisableAImodel = False
        for idx, key in enumerate(settingsKeys_new):
            if key == 'ai_model_library':
                modelLib = settings_new[idx]
                if modelLib is None or len(modelLib.strip()) == 0:
                    # no model library specified; disable AI model
                    forceDisableAImodel = True
                else:
                    if not modelLib in availableModels['prediction']:
                        raise Exception(f'Model library "{modelLib}" is not installed in this instance of AIDE.')
                    selectedModel = availableModels['prediction'][modelLib]
                    validAnnoTypes = ([selectedModel['annotationType']] if isinstance(selectedModel['annotationType'], str) else selectedModel['annotationType'])
                    validPredTypes = ([selectedModel['predictionType']] if isinstance(selectedModel['predictionType'], str) else selectedModel['predictionType'])
                    if not annoType in validAnnoTypes:
                        raise Exception(f'Model "{modelLib}" does not support annotations of type "{annoType}".')
                    if not predType in validPredTypes:
                        raise Exception(f'Model "{modelLib}" does not support predictions of type "{predType}".')
            
            elif key == 'ai_model_settings':
                # model settings are verified separately
                continue
                
            elif key == 'ai_alcriterion_library':
                modelLib = settings_new[idx]
                if modelLib is None or len(modelLib.strip()) == 0:
                    # no AL criterion library specified; disable AI model
                    forceDisableAImodel = True
                else:
                    if not modelLib in availableModels['ranking']:
                        raise Exception(f'Ranking library "{modelLib}" is not installed in this instance of AIDE.')
            
            elif key == 'ai_alcriterion_settings':
                # verify model parameters
                #TODO: outsource as well?
                pass

            elif key == 'segmentation_ignore_unlabeled':
                # only check if annotation type is segmentation mask
                if annoType == 'segmentationMasks' and settings_new[idx] is False:
                    # unlabeled areas are to be treated as "background": add class if not exists
                    addBackgroundClass = True

        if forceDisableAImodel:
            # switch flag
            flagFound = False
            for idx, key in enumerate(settingsKeys_new):
                if key == 'ai_model_enabled':
                    settings_new[idx] = False
                    flagFound = True
                    break
            if not flagFound:
                settings_new.append(False)
                settingsKeys_new.append('ai_model_enabled')

        # all checks passed; update database
        settings_new.append(project)
        queryStr = sql.SQL('''UPDATE aide_admin.project
            SET
            {}
            WHERE shortname = %s;
            '''
        ).format(
            sql.SQL(',').join([sql.SQL('{} = %s'.format(item)) for item in settingsKeys_new])
        )
        self.dbConn.execute(queryStr, tuple(settings_new), None)

        if addBackgroundClass:
            labelClasses = self.dbConn.execute(sql.SQL('''
                    SELECT * FROM {id_lc}
                ''').format(id_lc=sql.Identifier(project, 'labelclass')),
                None, 'all')
            hasBackground = False
            for lc in labelClasses:
                if lc['idx'] == 0:
                    hasBackground = True
                    break
            if not hasBackground:
                # find unique name
                lcNames = set([lc['name'] for lc in labelClasses])
                bgName = 'background'
                counter = 0
                while bgName in lcNames:
                    bgName = f'background ({counter})'
                    counter += 1
                self.dbConn.execute(sql.SQL('''
                    INSERT INTO {id_lc} (name, idx, hidden)
                    VALUES (%s, 0, true)
                ''').format(id_lc=sql.Identifier(project, 'labelclass')),
                (bgName,), None)

        response = {'status': 0}

        # check for and verify AI model settings
        if 'ai_model_settings' in settings:
            aiModelOptionsStatus = self.saveProjectModelSettings(project, settings['ai_model_settings'])
            response['ai_model_settings_status'] = aiModelOptionsStatus

        return response



    def listModelStates(self, project, latestOnly=False):
        modelLibraries = self.getAvailableAImodels()

        # get meta data about models shared through model marketplace
        result = self.dbConn.execute('''
            SELECT id, origin_uuid,
            author, anonymous, public,
            shared, tags, name, description,
            citation_info, license
            FROM aide_admin.modelMarketplace
            WHERE origin_project = %s OR origin_project IS NULL;
        ''', (project,), 'all')
        if result is not None and len(result):
            modelMarketplaceMeta = [] # {}
            for r in result:
                # mmID = r['id']
                values = {}
                for key in r.keys():
                    if isinstance(r[key], uuid.UUID):
                        values[key] = str(r[key])
                    else:
                        values[key] = r[key]
                modelMarketplaceMeta.append(values)
                # modelMarketplaceMeta[mmID] = values
        else:
            modelMarketplaceMeta = []   # {}

        # get project-specific model states
        if latestOnly:
            latestOnlyStr = sql.SQL('''
                WHERE timeCreated = (
                    SELECT MAX(timeCreated)
                    FROM {}
                )
            ''').format(sql.Identifier(project, 'cnnstate'))
        else:
            latestOnlyStr = sql.SQL('')

        queryStr = sql.SQL('''
            SELECT id, marketplace_origin_id, imported_from_marketplace,
                EXTRACT(epoch FROM timeCreated) AS time_created,
                model_library, alCriterion_library, num_pred, labelclass_autoupdate
            FROM (
                SELECT * FROM {id_cnnstate}
                {latestOnlyStr}
            ) AS cnnstate
            LEFT OUTER JOIN (
                SELECT cnnstate, COUNT(cnnstate) AS num_pred
                FROM {id_pred}
                GROUP BY cnnstate
            ) AS pred
            ON cnnstate.id = pred.cnnstate
            ORDER BY time_created DESC;
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate'),
            id_pred=sql.Identifier(project, 'prediction'),
            latestOnlyStr=latestOnlyStr
        )
        result = self.dbConn.execute(queryStr, None, 'all')
        response = []
        if result is not None and len(result):
            for r in result:
                mID = str(r['id'])
                try:
                    modelLibrary = modelLibraries['models']['prediction'][r['model_library']]
                except:
                    modelLibrary = {
                        'name': '(not found)'
                    }
                modelLibrary['id'] = r['model_library']
                try:
                    alCriterionLibrary = modelLibraries['models']['ranking'][r['alcriterion_library']]
                except:
                    alCriterionLibrary = {
                        'name': '(not found)'
                    }
                alCriterionLibrary['id'] = r['alcriterion_library']

                # Model Marketplace information
                marketplaceInfo = {}
                for mm in modelMarketplaceMeta:
                    if mID == mm['origin_uuid']:
                        # priority: model has been shared to Marketplace
                        marketplaceInfo = mm
                        break
                    elif str(r['marketplace_origin_id']) == mm['id'] and r['imported_from_marketplace']:
                        # model state comes from Marketplace
                        marketplaceInfo = mm

                # elif r['marketplace_origin_id'] in modelMarketplaceMeta:
                #     # model state comes from Marketplace
                #     marketplaceInfo = modelMarketplaceMeta[r['marketplace_origin_id']]
                # else:
                #     # model has no relationship to Marketplace
                #     marketplaceInfo = {}

                response.append({
                    'id': mID,
                    'time_created': r['time_created'],
                    'model_library': modelLibrary,
                    'al_criterion_library': alCriterionLibrary,
                    'num_pred': (r['num_pred'] if r['num_pred'] is not None else 0),
                    'labelclass_autoupdate': r['labelclass_autoupdate'],
                    'imported_from_marketplace': r['imported_from_marketplace'],
                    'marketplace_info': marketplaceInfo
                })
        return response



    def deleteModelStates(self, project, username, modelStateIDs):
        '''
            Receives a list of model state IDs (either str or UUID)
            and launches a task to delete them from the database.
            Unlike training and inference tasks, this one is routed
            through the default taskCoordinator.
        '''
        # verify IDs
        if not isinstance(modelStateIDs, Iterable):
            modelStateIDs = [modelStateIDs]
        modelStateIDs = [str(m) for m in modelStateIDs]
        process = aic_int.delete_model_states.s(project, modelStateIDs)
        taskID = self.taskCoordinator.submitJob(project, username, process, 'AIController')
        return taskID


    
    def duplicateModelState(self, project, username, modelStateID, skipIfLatest=True):
        '''
            Receives a model state ID and creates a copy of it in this project.
            This copy receives the current date, which makes it the most recent
            model state.
            If "skipIfLatest" is True and the model state with "modelStateID" is
            already the most recent state, no duplication is being performed.
        '''
        if not isinstance(modelStateID, uuid.UUID):
            modelStateID = uuid.UUID(modelStateID)
        
        process = aic_int.duplicate_model_state.s(project, modelStateID)
        taskID = self.taskCoordinator.submitJob(project, username, process, 'AIController')
        return taskID


    
    def getModelTrainingStatistics(self, project, username, modelStateIDs=None):
        '''
            Launches a task to assemble model-provided statistics
            into uniform series.
            Unlike training and inference tasks, this one is routed
            through the default taskCoordinator.
        '''
        # verify IDs
        if modelStateIDs is not None:
            try:
                if not isinstance(modelStateIDs, Iterable):
                    modelStateIDs = [modelStateIDs]
                modelStateIDs = [str(m) for m in modelStateIDs]
            except:
                modelStateIDs = None
        
        process = aic_int.get_model_training_statistics.s(project, modelStateIDs)
        taskID = self.taskCoordinator.submitJob(project, username, process, 'AIController')
        return taskID



    def getProjectModelSettings(self, project):
        '''
            Returns the AI and AL model properties for the given project,
            as stored in the database.
        '''
        result = self.dbConn.execute('''SELECT ai_model_enabled,
                ai_model_library, ai_model_settings,
                ai_alcriterion_library, ai_alcriterion_settings,
                numImages_autoTrain, minNumAnnoPerImage,
                maxNumImages_train, maxNumImages_inference
                FROM aide_admin.project
                WHERE shortname = %s;
            ''',
            (project,),
            1)
        return result[0]



    def saveProjectModelSettings(self, project, settings):
        # verify settings first
        optionsVerification = self.verifyAImodelOptions(project, settings)
        if optionsVerification['valid']:
            # save
            if isinstance(settings, dict):
                settings = json.dumps(settings)
            self.dbConn.execute('''
                UPDATE aide_admin.project
                SET ai_model_settings = %s
                WHERE shortname = %s;
            ''', (settings, project), None)
        else:
            optionsVerification['errors'].append('Model options have not passed verification and where therefore not saved.')
        return optionsVerification

    

    def getSavedWorkflows(self, project):
        queryStr = sql.SQL('''
            SELECT *
            FROM {id_workflow} AS wf
            LEFT OUTER JOIN (
                SELECT default_workflow
                FROM aide_admin.project
                WHERE shortname = %s
            ) AS defwf
            ON wf.id = defwf.default_workflow;
        ''').format(id_workflow=sql.Identifier(project, 'workflow'))
        result = self.dbConn.execute(queryStr, (project,), 'all')
        response = {}
        for r in result:
            response[str(r['id'])] = {
                'name': r['name'],
                'workflow': r['workflow'],
                'author': r['username'],
                'time_created': r['timecreated'].timestamp(),
                'time_modified': r['timemodified'].timestamp(),
                'default_workflow': (True if r['default_workflow'] is not None else False)
            }
        return response


    
    def saveWorkflow(self, project, username, workflow, workflowID, workflowName, setDefault=False):
        '''
            Receives a workflow definition (Python dict) to be saved
            in the database for a given project under a provided user
            name. The workflow definition is first parsed by the
            WorkflowDesigner and checked for validity. If it passes,
            it is stored in the database. If "setDefault" is True, the
            current workflow is set as the standard workflow, to be
            used for automated model training.
            Workflows can also be updated if an ID is specified.
        '''
        try:
            # check validity of workflow
            valid = self.workflowDesigner.parseWorkflow(project, workflow, verifyOnly=True)
            if not valid:
                raise Exception('Workflow is not valid.')   #TODO: detailed error message
            workflow = json.dumps(workflow)


            # check if workflow already exists
            queryArgs = [workflowName]
            if workflowID is not None:
                queryArgs.append(uuid.UUID(workflowID))
                idStr = sql.SQL(' OR id = %s')
            else:
                idStr = sql.SQL('')
            
            existingWorkflow = self.dbConn.execute(
                sql.SQL('''
                    SELECT id
                    FROM {id_workflow}
                    WHERE name = %s {idStr};
                ''').format(
                    id_workflow=sql.Identifier(project, 'workflow'),
                    idStr=idStr),
                tuple(queryArgs),
                1
            )
            if existingWorkflow is not None and len(existingWorkflow):
                existingWorkflow = existingWorkflow[0]['id']
            else:
                existingWorkflow = None

            # commit to database
            if existingWorkflow is not None:
                result = self.dbConn.execute(
                    sql.SQL('''
                        UPDATE {id_workflow}
                        SET name = %s, workflow = %s
                        WHERE id = %s
                        RETURNING id;
                    ''').format(id_workflow=sql.Identifier(project, 'workflow')),
                    (workflowName, workflow, existingWorkflow),
                    1
                )
            else:
                result = self.dbConn.execute(
                    sql.SQL('''
                        INSERT INTO {id_workflow} (name, workflow, username)
                        VALUES (%s, %s, %s)
                        RETURNING id;
                    ''').format(id_workflow=sql.Identifier(project, 'workflow')),
                    (workflowName, workflow, username),
                    1
                )
            wid = result[0]['id']

            # set as default if requested
            if setDefault:
                self.dbConn.execute(
                    '''
                        UPDATE aide_admin.project
                        SET default_workflow = %s
                        WHERE shortname = %s;
                    ''',
                    (wid, project,),
                    None
                )
            return {
                'status': 0,
                'id': str(wid)
            }
        except Exception as e:
            return {
                'status': 1,
                'message': str(e)
            }


    
    def setDefaultWorkflow(self, project, workflowID):
        '''
            Receives a str (workflow ID) and sets the associated
            workflow as default, if it exists for a given project.
        '''
        if isinstance(workflowID, str):
            workflowID = uuid.UUID(workflowID)
        if workflowID is not None and not isinstance(workflowID, uuid.UUID):
            return {
                'status': 2,
                'message': f'Provided argument "{str(workflowID)}" is not a valid workflow ID'
            }
        queryStr = sql.SQL('''
            UPDATE aide_admin.project
            SET default_workflow = (
                SELECT id FROM {id_workflow}
                WHERE id = %s
                LIMIT 1
            )
            WHERE shortname = %s
            RETURNING default_workflow;
        ''').format(
            id_workflow=sql.Identifier(project, 'workflow')
        )
        result = self.dbConn.execute(queryStr, (workflowID, project,), 1)
        if result is None or not len(result) or str(result[0]['default_workflow']) != str(workflowID):
            return {
                'status': 3,
                'message': f'Provided argument "{str(workflowID)}" is not a valid workflow ID'
            }
        else:
            return {
                'status': 0
            }


    
    def deleteWorkflow(self, project, username, workflowID):
        '''
            Receives a str or iterable of str variables under
            "workflowID" and finds and deletes them for a given
            project. Only deletes them if they were created by
            the user provided in "username", or if they are
            deleted by a super user.
        '''
        if isinstance(workflowID, str):
            workflowID = [uuid.UUID(workflowID)]
        elif not isinstance(workflowID, Iterable):
            workflowID = [workflowID]
        for w in range(len(workflowID)):
            if not isinstance(workflowID[w], uuid.UUID):
                workflowID[w] = uuid.UUID(workflowID[w])

        workflowID = tuple([(w,) for w in workflowID])

        queryStr = sql.SQL('''
            DELETE FROM {id_workflow}
            WHERE (
                username = %s
                OR username IN (
                    SELECT name
                    FROM aide_admin.user
                    WHERE isSuperUser = true
                )
            )
            AND id IN %s
            RETURNING id;
        ''').format(
            id_workflow=sql.Identifier(project, 'workflow')
        )
        result = self.dbConn.execute(queryStr, (username, workflowID,), 'all')
        result = [str(r['id']) for r in result]

        return {
            'status': 0,
            'workflowIDs': result
        }



    def deleteWorkflow_history(self, project, workflowID, revokeIfRunning=False):
        '''
            Receives a str or iterable of str variables under
            "workflowID" and finds and deletes them for a given
            project. Only deletes them if they were created by
            the user provided in "username", or if they are
            deleted by a super user.
            If "revokeIfRunning" is True, any workflow with ID
            given that is still running is aborted first.
        '''
        if workflowID == 'all':
            # get all workflow IDs from DB
            workflowID = self.dbConn.execute(
                sql.SQL('''
                    SELECT id FROM {id_workflowhistory};
                ''').format(
                    id_workflowhistory=sql.Identifier(project, 'workflowhistory')
                ),
                None,
                'all'
            )
            if workflowID is None or not len(workflowID):
                return {
                    'status': 0,
                    'workflowIDs': None
                }
            workflowID = [w['id'] for w in workflowID]

        elif isinstance(workflowID, str):
            workflowID = [uuid.UUID(workflowID)]
        elif not isinstance(workflowID, Iterable):
            workflowID = [workflowID]
        for w in range(len(workflowID)):
            if not isinstance(workflowID[w], uuid.UUID):
                workflowID[w] = uuid.UUID(workflowID[w])

        if not len(workflowID):
            return {
                'status': 0,
                'workflowIDs': None
            }

        if revokeIfRunning:
            WorkflowTracker._revoke_task([{'id': w} for w in workflowID])
        else:
            # skip ongoing tasks
            ongoingTasks = self.get_ongoing_tasks(project)
            for o in range(len(ongoingTasks)):
                if not isinstance(ongoingTasks[o], uuid.UUID):
                    ongoingTasks[o] = uuid.UUID(ongoingTasks[o])
            workflowID = list(set(workflowID).difference(set(ongoingTasks)))


        queryStr = sql.SQL('''
            DELETE FROM {id_workflowhistory}
            WHERE id IN %s
            RETURNING id;
        ''').format(
            id_workflowhistory=sql.Identifier(project, 'workflowhistory')
        )
        result = self.dbConn.execute(queryStr, (tuple(workflowID),), 'all')
        result = [str(r['id']) for r in result]

        return {
            'status': 0,
            'workflowIDs': result
        }


    
    def getLabelclassAutoadaptInfo(self, project, modelID=None):
        '''
            Retrieves information on whether the model in a project has been
            configured to automatically incorporate new classes by parameter
            expansion, as well as whether it is actually possible to disable
            the property (once enabled, it cannot be undone for any current
            model state).
            Also checks and returns whether AI model implementation actually
            supports label class adaptation.
        '''

        if modelID is not None:
            if not isinstance(modelID, uuid.UUID):
                modelID = uuid.UUID(modelID)
            modelIDstr = sql.SQL('WHERE id = %s')
            queryArgs = (modelID, project)
        else:
            modelIDstr = sql.SQL('''
                WHERE timeCreated = (
                    SELECT MAX(timeCreated)
                    FROM {id_cnnstate}
                )
            ''').format(
                id_cnnstate=sql.Identifier(project, 'cnnstate')
            )
            queryArgs = (project,)

        queryStr = sql.SQL('''
            SELECT 'model' AS row_type, labelclass_autoupdate, NULL AS ai_model_library
            FROM {id_cnnstate}
            {modelIDstr}
            UNION ALL
            SELECT 'project' AS row_type, labelclass_autoupdate, ai_model_library
            FROM "aide_admin".project
            WHERE shortname = %s;
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate'),
            modelIDstr=modelIDstr
        )
        result = self.dbConn.execute(queryStr, queryArgs, 2)
        response = {
            'model': False,
            'model_lib': False,
            'project': False
        }
        for row in result:
            if row['ai_model_library'] is not None:
                # check if AI model library supports adaptation
                modelLib = row['ai_model_library']
                response['model_lib'] = self.aiModels['prediction'][modelLib]['canAddLabelclasses']
            response[row['row_type']] = row['labelclass_autoupdate']
        
        return response



    def setLabelclassAutoadaptEnabled(self, project, enabled):
        '''
            Sets automatic labelclass adaptation to the specified value.
            This is only allowed if the current model state does not already
            have automatic labelclass adaptation enabled.
        '''
        if not enabled:
            # user requests to disable adaptation; check if possible
            enabled_model = self.getLabelclassAutoadaptInfo(project, None)
            if enabled_model['model']:
                # current model has adaptation enabled; abort
                return False
        
        result = self.dbConn.execute('''
            UPDATE "aide_admin".project
            SET labelclass_autoupdate = %s
            WHERE shortname = %s
            RETURNING labelclass_autoupdate;
        ''', (enabled, project), None)

        return result[0]['labelclass_autoupdate']