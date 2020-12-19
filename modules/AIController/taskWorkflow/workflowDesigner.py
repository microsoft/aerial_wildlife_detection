'''
    The workflow designer allows users to create and launch
    sophisticated chains of (distributed) training and inference,
    including customization of all available hyperparameters.

    To this end, workflows can be submitted as a Python dict
    object as follows:

        {
            "project": "project_shortname",
            "tasks": [
                {
                    "id": "node0",
                    "type": "train"
                },
                {
                    "id": "node1",
                    "type": "train",
                    "kwargs": {
                        "includeGoldenQuestions": False,
                        "minNumAnnotations": 1
                    }
                },
                "inference"
            ],
            "repeaters": {
                "repeater0": {
                    "id": "repeater0",
                    "type": "repeater",
                    "start_node": "node1",
                    "end_node": "node0",
                    "kwargs": {
                        "num_repetitions": 2
                    }
                }
            },
            "options": {
                "max_num_workers": 3
            }
        }
    
    Note the following:
        - The sequence of tasks to be executed is the order of the
          entries in the list under "tasks."
        - Tasks may be specified as simple names ("train", "inference").
          In this case, parameters will be taken from the global "options"
          provided, or else from the default options if missing (see file
          "defaultOptions.py").
        - Tasks can also be specified as sub-dicts with name and parameters
          under "kwargs." Those have priority over global options, but as
          above, global options or else defaults will be used to auto-complete
          missing options, if necessary.
        - Special repeater entries specify a sequence of nodes via "start_node"
          and "end_node" that is to be looped for "num_repetitions" times. For
          repeater entries to work, the start and end nodes of the loop must be
          specified as a dict and contain an id.
        - Repeater nodes may also have the same id for "start_node" and "end_node,"
          which results in a single task being executed "num_repetitions" times.

    2020 Benjamin Kellenberger
'''

from psycopg2 import sql
import celery

from .defaultOptions import DEFAULT_WORKFLOW_ARGS
from modules.AIController.backend import celery_interface as aic_int
from modules.AIWorker.backend import celery_interface as aiw_int



class WorkflowDesigner:

    def __init__(self, dbConnector, celeryApp):
        self.dbConnector = dbConnector
        self.celeryApp = celeryApp


    def _get_num_available_workers(self):
        #TODO: improve...
        numWorkers = 0
        i = self.celeryApp.control.inspect()
        if i is not None:
            activeQueues = i.active_queues()
            if activeQueues is not None:
                for qName in activeQueues.keys():
                    queue = activeQueues[qName]
                    for subqueue in queue:
                        if 'name' in subqueue and subqueue['name'] == 'AIWorker':
                            numWorkers += 1
        return numWorkers


    def _get_project_defaults(self, project):
        '''
            Queries and returns default values for some project-specific
            parameters.
        '''
        queryStr = sql.SQL('''
            SELECT minNumAnnoPerImage, maxNumImages_train, maxNumImages_inference
            FROM aide_admin.project
            WHERE shortname = %s;
        ''')
        result = self.dbConnector.execute(queryStr, (project,), 1)
        result = result[0]
        return {
            'train': {
                'min_anno_per_image': result['minnumannoperimage'],
                'max_num_images': result['maxnumimages_train']
            },
            'inference': {
                'max_num_images': result['maxnumimages_inference']
            }
        }


    def _expand_from_name(self, index, project, taskName, workflow, projDefaults):
        '''
            Creates and returns a task description dict from a task name.
            Receives the workflow description for global arguments, but also
            resorts to default arguments for auto-completion, if necessary.
        '''
        if not taskName in DEFAULT_WORKFLOW_ARGS:
            raise Exception(f'Unknown task name provided ("{taskName}") for task at index {index}.')

        # default arguments
        taskArgs = DEFAULT_WORKFLOW_ARGS[taskName]

        # replace with global options if available
        for key in taskArgs.keys():
            if key in workflow['options']:
                taskArgs[key] = workflow['options'][key]    #TODO: sanity checks (type, values, etc.)
            elif key in projDefaults[taskName]:
                taskArgs[key] = projDefaults[taskName][key]

        return {
            'type': taskName,
            'project': project,
            'kwargs': taskArgs
        }

    
    def _get_training_signature(self, project, taskArgs, isFirstNode=False):
        epoch = taskArgs['epoch']
        numEpochs = taskArgs['numEpochs']
        numWorkers = taskArgs['max_num_workers']
        aiModelSettings = (taskArgs['ai_model_settings'] if 'ai_model_settings' in taskArgs else None)

        # initialize list for Celery chain tasks
        taskList = []

        if not 'data' in taskArgs:
            # no list of images provided; prepend getting training images
            minNumAnnoPerImage = taskArgs['min_anno_per_image']
            if isinstance(minNumAnnoPerImage, str):
                if len(minNumAnnoPerImage):
                    minNumAnnoPerImage = int(minNumAnnoPerImage)
                else:
                    minNumAnnoPerImage = None

            maxNumImages = taskArgs['max_num_images']
            if isinstance(maxNumImages, str):
                if len(maxNumImages):
                    maxNumImages = int(maxNumImages)
                else:
                    maxNumImages = None

            img_task_kwargs = {'project': project,
                            'epoch': epoch,
                            'numEpochs': numEpochs,
                            'minTimestamp': taskArgs['min_timestamp'],
                            'includeGoldenQuestions': taskArgs['include_golden_questions'],
                            'minNumAnnoPerImage': minNumAnnoPerImage,
                            'maxNumImages': maxNumImages,
                            'numWorkers': numWorkers}
            if isFirstNode:
                # first node: prepend update model task and fill blank
                img_task_kwargs['blank'] = None
                update_model_kwargs = {'project': project,
                                    'numEpochs': numEpochs,
                                    'blank': None}
                taskList.append(
                    celery.group([
                        aic_int.get_training_images.s(**img_task_kwargs).set(queue='AIController'),
                        aiw_int.call_update_model.s(**update_model_kwargs).set(queue='AIWorker')
                    ])
                )
            else:
                taskList.append(
                    aic_int.get_training_images.s(**img_task_kwargs).set(queue='AIController')
                )

            trainArgs = {
                'epoch': epoch,
                'numEpochs': numEpochs,
                'project': project,
                'aiModelSettings': aiModelSettings
            }
        
        else:
            trainArgs = {
                'data': taskArgs['data'],
                'epoch': epoch,
                'numEpochs': numEpochs,
                'project': project,
                'aiModelSettings': aiModelSettings
            }
        

        if numWorkers > 1:
            # distribute training; also need to call model state averaging
            trainTasks = []
            for w in range(numWorkers):
                train_kwargs = {**trainArgs, **{'index':w}}
                trainTasks.append(aiw_int.call_train.s(**train_kwargs).set(queue='AIWorker'))
            taskList.append(
                celery.chord(
                    trainTasks,
                    aiw_int.call_average_model_states.si(**{'epoch':epoch, 'numEpochs':numEpochs, 'project':project, 'aiModelSettings':aiModelSettings}).set(queue='AIWorker')
                )
            )
        
        else:
            # training on single worker
            train_kwargs = {**trainArgs, **{'index':0}}
            taskList.append(
                aiw_int.call_train.s(**train_kwargs).set(queue='AIWorker')
            )
        return celery.chain(taskList)


    def _get_inference_signature(self, project, taskArgs, isFirstNode=False):
        epoch = taskArgs['epoch']
        numEpochs = taskArgs['numEpochs']
        numWorkers = taskArgs['max_num_workers']
        maxNumImages = taskArgs['max_num_images']
        if isinstance(maxNumImages, str):
            if len(maxNumImages):
                maxNumImages = int(maxNumImages)
            else:
                maxNumImages = None
        aiModelSettings = (taskArgs['ai_model_settings'] if 'ai_model_settings' in taskArgs else None)
        alCriterionSettings = (taskArgs['alcriterion_settings'] if 'alcriterion_settings' in taskArgs else None)

        # initialize list for Celery chain tasks
        taskList = []

        if not 'data' in taskArgs:
            # no list of images provided; prepend getting inference images
            img_task_kwargs = {'project': project,
                            'epoch': epoch,
                            'numEpochs': numEpochs,
                            'goldenQuestionsOnly': taskArgs['golden_questions_only'],
                            'maxNumImages': maxNumImages,
                            'numWorkers': numWorkers}
            if isFirstNode:
                # first task to be executed; prepend model update and fill blanks
                img_task_kwargs['blank'] = None
                update_model_kwargs = {'project': project,
                                    'numEpochs': numEpochs,
                                    'blank': None}
                taskList.append(
                    celery.group([
                        aic_int.get_inference_images.s(**img_task_kwargs).set(queue='AIController'),
                        aiw_int.call_update_model.s(**update_model_kwargs).set(queue='AIWorker')
                    ])
                )
            else:
                taskList.append(
                    aic_int.get_inference_images.s(**img_task_kwargs).set(queue='AIController')
                )

            inferenceArgs = {
                'epoch': epoch,
                'numEpochs': numEpochs,
                'project': project,
                'aiModelSettings': aiModelSettings,
                'alCriterionSettings': alCriterionSettings
            }
        
        else:
            inferenceArgs = {
                'data': taskArgs['data'],
                'epoch': epoch,
                'numEpochs': numEpochs,
                'project': project,
                'aiModelSettings': aiModelSettings,
                'alCriterionSettings': alCriterionSettings
            }

        if numWorkers > 1:
            # distribute inference
            inferenceTasks = []
            for w in range(numWorkers):
                inference_kwargs = {**inferenceArgs, **{'index':w}}
                inferenceTasks.append(aiw_int.call_inference.s(**inference_kwargs).set(queue='AIWorker'))
            taskList.append(celery.group(inferenceTasks))
        
        else:
            # training on single worker
            inference_kwargs = {**inferenceArgs, **{'index':0}}
            taskList.append(
                aiw_int.call_inference.s(**inference_kwargs).set(queue='AIWorker')
            )
        return celery.chain(taskList)


    def _create_celery_task(self, project, taskDesc, isFirstTask, verifyOnly=False):
        '''
            Receives a task description (full dict with name and kwargs)
            and creates true Celery task routines from it.
            If "verifyOnly" is set to True, it just returns a bool indi-
            cating whether the task description is valid (True) or not
            (False).
            Accounts for special cases, such as:
                - train: if more than one worker is specified, the task is
                         a chain of distributed training and model state
                         averaging.
                - train and inference: if no list of image IDs is provided,
                                       a job of retrieving the latest set of
                                       images is prepended.
                - etc.
            
            Returns a Celery job that can be appended to a global chain.
        '''
        try:
            taskName = taskDesc['type'].lower()
            if taskName == 'train':
                task = self._get_training_signature(project, taskDesc['kwargs'], isFirstTask)
            elif taskName == 'inference':
                task = self._get_inference_signature(project, taskDesc['kwargs'], isFirstTask)
            else:
                task = None
        except:
            task = None
        if verifyOnly:
            return (task is not None)
        else:
            return task


    def parseWorkflow(self, project, workflow, verifyOnly=False):
        '''
            Parses a workflow as described in the header of this file. Auto-
            completes missing arguments and provides appropriate function ex-
            pansion wherever needed (e.g., "train" may become "get images" > 
            "train across multiple workers" > "average model states").

            If "verifyOnly" is set to True, the function returns a bool indi-
            cating whether the workflow is valid (True) or not (False).
            Else, it returns a Celery chain that can be submitted to the task
            queue via the AIController's middleware.
        '''

        #TODO: sanity checks
        if not 'options' in workflow:
            workflow['options'] = {}    # for compatibility

        # get number of available workers
        numWorkersMax = self._get_num_available_workers()

        # get default project settings for some of the parameters
        projDefaults = self._get_project_defaults(project)

        # expand task specifications with repeaters
        workflow_expanded = workflow['tasks']
        if 'repeaters' in workflow:
            # get node order first
            nodeOrder = []
            nodeIndex = {}
            for idx, node in enumerate(workflow_expanded):
                if isinstance(node, dict) and 'id' in node:
                    nodeOrder.append(node['id'])
                    nodeIndex[node['id']] = idx

            # get start node for repeaters
            startNodeIDs = {}
            for key in workflow['repeaters']:
                startNode = workflow['repeaters'][key]['start_node']
                startNodeIDs[startNode] = key
            
            # process repeaters front to back (start with first)
            for nodeID in nodeOrder:
                if nodeID in startNodeIDs:
                    # find indices of start and end node
                    startNodeIndex = nodeIndex[nodeID]
                    repeaterID = startNodeIDs[nodeID]
                    endNodeIndex = nodeIndex[workflow['repeaters'][repeaterID]['end_node']]

                    # extract and expand sub-workflow
                    subWorkflow = workflow['tasks'][endNodeIndex:startNodeIndex+1]
                    targetSubWorkflow = []
                    numRepetitions = workflow['repeaters'][repeaterID]['kwargs']['num_repetitions']
                    for _ in range(numRepetitions):
                        targetSubWorkflow.extend(subWorkflow.copy())

                    # insert after
                    workflow_expanded = workflow_expanded[:startNodeIndex+1] + targetSubWorkflow + workflow_expanded[startNodeIndex+1:]
        
        # epoch counter (only training jobs can increment it)
        epoch = 1

        # parse entries in workflow
        taskDescriptions = []
        for index, taskSpec in enumerate(workflow_expanded):
            if isinstance(taskSpec, str):
                # task name provided
                if taskSpec == 'repeater' or taskSpec == 'connector':
                    continue
                #auto-expand into dict first
                taskDesc = self._expand_from_name(index, project, taskSpec, workflow, projDefaults)
                taskName = taskDesc['type']

            elif isinstance(taskSpec, dict):
                # task dictionary provided; verify and auto-complete if necessary
                taskDesc = taskSpec.copy()
                if not 'type' in taskDesc:
                    raise Exception(f'Task at index {index} is of unknown type.')
                taskName = taskDesc['type']
                if taskName == 'repeater' or taskName == 'connector':
                    continue
                if not taskName in DEFAULT_WORKFLOW_ARGS:
                    raise Exception(f'Unknown task type provided ("{taskName}") for task at index {index}.')
                
            defaultArgs = DEFAULT_WORKFLOW_ARGS[taskName].copy()
            if not 'kwargs' in taskDesc:
                # no arguments provided; add defaults
                taskDesc['kwargs'] = defaultArgs

                # replace with global arguments wherever possible
                for key in taskDesc['kwargs'].keys():
                    if key in workflow['options']:
                        taskDesc['kwargs'][key] = workflow['options'][key]
                    elif key in projDefaults[taskName]:
                        taskDesc['kwargs'][key] = projDefaults[taskName][key]
            
            else:
                # arguments provided; auto-complete wherever needed
                for key in defaultArgs.keys():
                    if not key in taskDesc['kwargs']:
                        if key in workflow['options']:
                            # global option available
                            taskDesc['kwargs'][key] = workflow['options'][key]
                        elif key in projDefaults[taskName]:
                            # fallback 1: default project setting
                            taskDesc['kwargs'][key] = projDefaults[taskName][key]
                        else:
                            # fallback 2: default option
                            taskDesc['kwargs'][key] = defaultArgs[key]

            if 'max_num_workers' in taskDesc['kwargs']:
                if isinstance(taskDesc['kwargs']['max_num_workers'], str):
                    if len(taskDesc['kwargs']['max_num_workers']):
                        taskDesc['kwargs']['max_num_workers'] = int(taskDesc['kwargs']['max_num_workers'])
                    else:
                        taskDesc['kwargs']['max_num_workers'] = defaultArgs['max_num_workers']
                
                taskDesc['kwargs']['max_num_workers'] = min(
                    taskDesc['kwargs']['max_num_workers'],
                    numWorkersMax
                )
            else:
                taskDesc['kwargs']['max_num_workers'] = defaultArgs['max_num_workers']

            taskDesc['kwargs']['epoch'] = epoch
            if taskName.lower() == 'train':
                epoch += 1

            taskDescriptions.append(taskDesc)

        # construct celery tasks out of descriptions
        tasklist = []
        for index, taskDesc in enumerate(taskDescriptions):
            # add number of epochs as argument
            taskDesc['kwargs']['numEpochs'] = epoch
            task = self._create_celery_task(project, taskDesc, isFirstTask=(True if index==0 else False), verifyOnly=verifyOnly)
            tasklist.append(task)

        if verifyOnly:
            #TODO: detailed warnings and errors
            return all(tasklist)
        
        else:
            chain = celery.chain(tasklist)
            return chain