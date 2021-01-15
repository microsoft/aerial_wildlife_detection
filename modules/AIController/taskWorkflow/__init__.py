'''
    2020-21 Benjamin Kellenberger
'''

def task_ids_match(workflow, taskID):
    '''
        Iterates through all the subtasks in a given workflow
        and compares their IDs with the given taskID. Returns
        True if one of them matches and False otherwise.
    '''
    if isinstance(workflow, list):
        for wf in workflow:
            if task_ids_match(wf, taskID):
                return True
    elif isinstance(workflow, dict):
        if 'id' in workflow:
            if workflow['id'] == taskID:
                return True
        if 'children' in workflow:
            if task_ids_match(workflow['children'], taskID):
                return True
    elif isinstance(workflow, str):
        if workflow == taskID:
            return True
    return False