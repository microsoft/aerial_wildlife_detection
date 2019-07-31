'''
    Miscellaneous helper functions.

    2019 Benjamin Kellenberger
'''

import importlib
from datetime import datetime
import pytz



def array_split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr = arr[size:]
     arrs.append(arr)
     return arrs



def current_time():
    '''
        Returns the current time with UTC timestamp.
    '''
    return datetime.now(tz=pytz.utc)



def get_class_executable(path):
    '''
        Loads a Python class path (e.g. package.subpackage.classFile.className)
        and returns the class executable (under 'className').
    '''
    
    # split classPath into path and executable name
    idx = path.rfind('.')
    classPath, executableName = path[0:idx], path[idx+1:]
    execFile = importlib.import_module(classPath)
    return getattr(execFile, executableName)


def check_args(options, defaultOptions):
    '''
        Compares a dictionary of objects ('options') with a set of 'defaultOptions'
        options and copies entries from the default set to the provided options
        if they are not in there.
    '''
    def __check(options, default):
        if not isinstance(default, dict):
            return options
        for key in default.keys():
            if not key in options:
                options[key] = default[key]
            if not key == 'transform':
                options[key] = __check(options[key], default[key])
        return options
    if options is None or not isinstance(options, dict):
        return defaultOptions
    else:
        return __check(options, defaultOptions)