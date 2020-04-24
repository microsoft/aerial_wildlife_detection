'''
    Contains keyword arguments and their default values
    for workflows. Used to auto-complete arguments that
    are missing in submitted workflows.
    Workflow items with all the given arguments are then
    parsed into actual Celery workflows by the workflow
    designer.

    2020 Benjamin Kellenberger
'''

DEFAULT_WORKFLOW_ARGS = {
    'train': {
        'min_timestamp': 'lastState',
        'min_anno_per_image': 0,
        'include_golden_questions': True,   #TODO
        'max_num_images': -1,
        'max_num_workers': -1
    },
    'inference': {
        'force_unlabeled': False,       #TODO
        'golden_questions_only': False, #TODO
        'max_num_images': -1,
        'max_num_workers': -1
    },
    #TODO
}