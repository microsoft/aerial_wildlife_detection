'''
    Miscellaneous helper functions.

    2019-20 Benjamin Kellenberger
'''

import os
import importlib
from functools import reduce
from datetime import datetime
import pytz
import socket
from urllib.parse import urlsplit
import netifaces
import html
import base64
import numpy as np
from PIL import Image, ImageColor
from psycopg2 import sql



class LogDecorator:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def print_status(status, color=None):
        if status.lower() == 'ok':
            print(f'{LogDecorator.OKGREEN}[ OK ]{LogDecorator.ENDC}'.ljust(30))
        elif status.lower() == 'warn':
            print(f'{LogDecorator.WARNING}[WARN]{LogDecorator.ENDC}'.ljust(30))
        elif status.lower() == 'fail':
            print(f'{LogDecorator.FAIL}[FAIL]{LogDecorator.ENDC}'.ljust(30))
        else:
            if color is not None:
                print(f'{getattr(LogDecorator, color)}[{status}]{LogDecorator.ENDC}'.ljust(30))
            else:
                print(f'[{status}]'.ljust(30))


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



def parse_boolean(boolean):
    '''
        Performs parsing of various specifications of a boolean:
        True, 'True', 'true', 't', 'yes', '1', etc.
    '''
    if isinstance(boolean, bool):
        return boolean
    elif isinstance(boolean, str):
        boolean = boolean.lower()
        if boolean.startswith('t') or boolean == '1' or boolean.startswith('y'):
            return True
        else:
            return False


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
            value = html.escape(value)
        value = dataType(value)
        outputVals.append(value)
        outputKeys.append(nextKey)
    return outputVals, outputKeys



def checkDemoMode(project, dbConnector):
        '''
            Returns a bool indicating whether the project is in demo mode.
            Returns None if the project does not exist.
        '''
        try:
            response = dbConnector.execute('''
                SELECT demoMode FROM aide_admin.project
                WHERE shortname = %s;''',
                (project,),
                1)
            if len(response):
                return response[0]['demomode']
            else:
                return None
        except:
            return None
        


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



def is_localhost(baseURI):
    '''
        Receives a URI and checks whether it points to this
        host ("localhost", "/", same socket name, etc.).
        Returns True if it is the same machine, and False
        otherwise.
    '''
    # check for explicit localhost or hostname appearance in URL
    localhosts = ['localhost', socket.gethostname()]
    interfaces = netifaces.interfaces()
    for i in interfaces:
        iface = netifaces.ifaddresses(i).get(netifaces.AF_INET)
        if iface != None:
            for j in iface:
                localhosts.append(j['addr'])
    
    baseURI_fragments = urlsplit(baseURI)
    baseURI_stripped = baseURI_fragments.netloc
    for l in localhosts:
        if baseURI_stripped.startswith(l):
            return True
    
    # also check for local addresses that do not even specify the hostname (e.g. '/files' or just 'files')
    if not baseURI.startswith('http'):
        return True
    
    # all checks failed; file server is running on another machine
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



def hexToRGB(hexString):
    '''
        Receives a HTML/CSS-compliant hex color string
        of one of the following formats (hash symbol optional):
            "#RRGGBB"
            "#RGB"
        and returns a tuple of (Red, Green, Blue) values in the
        range of [0, 255].
    '''
    assert isinstance(hexString, str), f'ERROR: "{str(hexString)}" is not a valid string.'
    if not hexString.startswith('#'):
        hexString = '#' + hexString
    assert len(hexString)>1, f'ERROR: the provided string is empty.'

    return ImageColor.getrgb(hexString)



def imageToBase64(image):
    '''
        Receives a PIL image and converts it
        into a base64 string.
        Returns that string plus the width
        and height of the image.
    '''
    dataArray = np.array(image).astype(np.uint8)
    b64str = base64.b64encode(dataArray.ravel()).decode('utf-8')
    return b64str, image.width, image.height



def base64ToImage(base64str, width, height, toPIL=True):
    '''
        Receives a base64-encoded string as stored in
        AIDE's database (e.g. for segmentation masks) and
        returns a PIL image with its contents if "toPIL" is
        True (default), or an ndarray otherwise.
    '''
    raster = np.frombuffer(base64.b64decode(base64str), dtype=np.uint8)
    raster = np.reshape(raster, (int(height),int(width),))
    if not toPIL:
        return raster
    image = Image.fromarray(raster)
    return image



def setImageCorrupt(dbConnector, project, imageID, corrupt):
    '''
        Sets the "corrupt" flag to the provided value for a
        given project and image ID.
    '''
    queryStr = sql.SQL('''
            UPDATE {id_img}
            SET corrupt = %s
            WHERE id = %s;
        ''').format(
            id_img=sql.Identifier(project, 'image')
        )
    dbConnector.execute(queryStr, (corrupt,imageID,), None)



def getPILimage(input, imageID, project, dbConnector, convertRGB=False):
    '''
        Reads an input (file path or BytesIO object) and
        returns a PIL image instance.
        Also checks if the image is intact. If it is not,
        the "corrupt" flag is set in the database as True,
        and None is returned.
    '''
    img = None
    try:
        img = Image.open(input)
        if convertRGB:
            # conversion implicitly verifies the image (TODO)
            img = img.convert('RGB')
        else:
            img.verify()
            img = Image.open(input)

    except:
        # something failed; set "corrupt" flag to False for image
        setImageCorrupt(dbConnector, project, imageID, True)
    
    finally:
        return img



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