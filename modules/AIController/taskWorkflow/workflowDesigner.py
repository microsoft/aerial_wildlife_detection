'''
    The workflow designer allows users to create and launch
    sophisticated chains of (distributed) training and inference,
    including customization of all available hyperparameters.

    To this end, workflows can be submitted as a Python dict
    object as follows:

        {
            "project": "project_shortname",
            "tasks": [
                "train",
                "inference",
                {
                    "name": "train",
                    "kwargs": {
                        "includeGoldenQuestions": False,
                        "minNumAnnotations": 1
                    }
                },
                "inference"
            ],
            "options": {
                "max_num_workers": 3,
                "include_golden_TODO": True,
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
            # stats = i.stats()
            # if stats is not None:
            #     #TODO: filter according to queues
            #     return len(i.stats())
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
            'name': taskName,
            'project': project,
            'kwargs': taskArgs
        }

    
    def _get_training_signature(self, project, taskArgs):
        epoch = taskArgs['epoch']
        numWorkers = taskArgs['max_num_workers']

        # initialize list for Celery chain tasks
        taskList = []

        if not 'data' in taskArgs:
            # no list of images provided; prepend getting training images
            taskList.append(
                aic_int.get_training_images.s(**{'project': project,
                                                'epoch': epoch,
                                                'minTimestamp': taskArgs['min_timestamp'],
                                                'includeGoldenQuestions': taskArgs['include_golden_questions'],
                                                'minNumAnnoPerImage': taskArgs['min_anno_per_image'],
                                                'maxNumImages': taskArgs['max_num_images'],
                                                'numWorkers': numWorkers}).set(queue='AIController')
            )
            trainArgs = {
                'epoch': epoch,
                'project': project
            }
        
        else:
            trainArgs = {
                'data': taskArgs['data'],
                'epoch': epoch,
                'project': project
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
                    aiw_int.call_average_model_states.si(**{'epoch':epoch, 'project':project}).set(queue='AIWorker')
                )
            )
        
        else:
            # training on single worker
            train_kwargs = {**trainArgs, **{'index':0}}
            taskList.append(
                aiw_int.call_train.s(**train_kwargs).set(queue='AIWorker')
            )
        return celery.chain(taskList)


    def _get_inference_signature(self, project, taskArgs):
        epoch = taskArgs['epoch']
        numWorkers = taskArgs['max_num_workers']

        # initialize list for Celery chain tasks
        taskList = []

        if not 'data' in taskArgs:
            # no list of images provided; prepend getting inference images
            taskList.append(
                aic_int.get_inference_images.s(**{'project': project,
                                                'epoch': epoch,
                                                'goldenQuestionsOnly': taskArgs['golden_questions_only'],
                                                'maxNumImages': taskArgs['max_num_images'],
                                                'numWorkers': numWorkers}).set(queue='AIController')
            )
            inferenceArgs = {
                'epoch': epoch,
                'project': project
            }
        
        else:
            inferenceArgs = {
                'data': taskArgs['data'],
                'epoch': epoch,
                'project': project
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


    def _create_celery_task(self, taskDesc):
        '''
            Receives a task description (full dict with name and kwargs)
            and creates true Celery task routines from it.
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
        taskName = taskDesc['name'].lower()
        project = taskDesc['project']
        if taskName == 'train':
            task = self._get_training_signature(project, taskDesc['kwargs'])
        elif taskName == 'inference':
            task = self._get_inference_signature(project, taskDesc['kwargs'])
        return task


    def parseWorkflow(self, project, workflow):
        '''
            Parses a workflow as described in the header of this file. Auto-
            completes missing arguments and provides appropriate function ex-
            pansion wherever needed (e.g., "train" may become "get images" > 
            "train across multiple workers" > "average model states").

            Returns a Celery chain that can be submitted to the task queue via
            the AIController's middleware.
        '''

        #TODO: sanity checks
        if not 'options' in workflow:
            workflow['options'] = {}    # for compatibility

        # get number of available workers
        numWorkersMax = self._get_num_available_workers()

        # get default project settings for some of the parameters
        projDefaults = self._get_project_defaults(project)

        # initialize list for Celery chain tasks
        tasklist = []
        
        # epoch counter (only training jobs can increment it)
        epoch = 1

        # parse entries in workflow
        for index, taskSpec in enumerate(workflow['tasks']):
            if isinstance(taskSpec, str):
                # project name provided; auto-expand into dict first
                taskDesc = self._expand_from_name(index, project, taskSpec, workflow, projDefaults)
                taskName = taskDesc['name']

            elif isinstance(taskSpec, dict):
                # task dictionary provided; verify and auto-complete if necessary
                taskDesc = taskSpec.copy()
                if not 'name' in taskDesc:
                    raise Exception(f'Task at index {index} is unnamed.')
                taskName = taskDesc['name']
                if not taskName in DEFAULT_WORKFLOW_ARGS:
                    raise Exception(f'Unknown task name provided ("{taskName}") for task at index {index}.')
                
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
                taskDesc['kwargs']['max_num_workers'] = min(
                    taskDesc['kwargs']['max_num_workers'],
                    numWorkersMax
                )

            taskDesc['kwargs']['epoch'] = epoch
            if taskName.lower() == 'train':
                epoch += 1

            # construct celery task out of description
            task = self._create_celery_task(taskDesc)
            tasklist.append(task)

        chain = celery.chain(tasklist)
        return chain