'''
    Miscellaneous helper functions.

    2019-20 Benjamin Kellenberger
'''

import os
import importlib
from datetime import datetime
import pytz
import cgi



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


def parse_parameters(data, params, absent_ok=True, escape=True):
    '''
        Accepts a dict (data) and list (params) and assembles
        an output list of the entries in data under keys of params, in order of
        the latter.
        If params is a list of lists, the first entry of the nested list is the
        key, and the second the data type to which the value will be cast.
        Raises an Exception if typecasting fails.
        If absent_ok is True, missing keys will be skipped.
        If escape is True, sensitive characters will be escaped from strings.

        Also returns a list of the keys that were eventually added.
    '''
    outputVals = []
    outputKeys = []
    for idx in range(len(params)):
        if isinstance(params[idx], str):
            nextKey = params[idx]
            dataType = str
        else:
            nextKey = params[idx][0]
            dataType = params[idx][1]
        
        if not nextKey in data and absent_ok:
            continue

        value = data[nextKey]
        if escape and isinstance(value, str):
            value = cgi.escape(value)
        value = dataType(value)
        outputVals.append(value)
        outputKeys.append(nextKey)
    return outputVals, outputKeys
        

def is_fileServer(config):
    '''
        Returns True if the current instance is a valid
        file server. For this, the following two properties
        must hold:
        - the "staticfiles_dir" property under "[FileServer]"
          in the configuration.ini file must be a valid directory
          on this machine;
        - environment variable "AIDE_MODULES" must be set to contain
          the string "FileServer" (check is performed without case)
    '''

    try:
        return ('fileserver' in os.environ['AIDE_MODULES'].lower() and \
            os.path.isdir(config.getProperty('FileServer', 'staticfiles_dir'))
        )
    except:
       return False






def listDirectory(baseDir, recursive=False):
    '''
        Similar to glob's recursive file listing, but
        implemented so that circular softlinks are avoided.
        Removes the baseDir part (with trailing separator)
        from the files returned.
    '''
    files_disk = set()
    if not baseDir.endswith(os.sep):
        baseDir += os.sep
    def _scan_recursively(imgs, baseDir, fileDir, recursive):
        files = os.listdir(fileDir)
        for f in files:
            path = os.path.join(fileDir, f)
            if os.path.isfile(path) and os.path.splitext(f)[1].lower() in valid_image_extensions:
                imgs.add(path)
            elif os.path.islink(path):
                if os.readlink(path) in baseDir:
                    # circular link; avoid
                    continue
                elif recursive:
                    imgs = _scan_recursively(imgs, baseDir, path, True)
            elif os.path.isdir(path) and recursive:
                imgs = _scan_recursively(imgs, baseDir, path, True)
        return imgs

    files_scanned = _scan_recursively(set(), baseDir, baseDir, recursive)
    for f in files_scanned:
        files_disk.add(f.replace(baseDir, ''))
    return files_disk


valid_image_extensions = (
    '.jpg',
    '.jpeg',
    '.png',
    '.gif',
    # '.tif',
    # '.tiff',
    '.bmp',
    '.ico',
    '.jfif',
    '.pjpeg',
    '.pjp'
)


valid_image_mime_types = (
    'image/jpeg',
    'image/bmp',
    'image/x-windows-bmp',
    'image/gif',
    'image/x-icon',
    'image/png'
)