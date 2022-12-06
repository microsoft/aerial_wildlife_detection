'''
    Miscellaneous helper functions.

    2019-22 Benjamin Kellenberger
'''

import os
import sys
import importlib
import unicodedata
import random
import re
from collections.abc import Iterable
from datetime import datetime
import pytz
import socket
from urllib.parse import urlsplit
import netifaces
import requests
import html
import base64
import numpy as np
from PIL import Image, ImageColor
from psycopg2 import sql

from util.logDecorator import LogDecorator
from util import drivers


def toNumber(value):
    if isinstance(value, int) or isinstance(value, float):
        return value
    elif isinstance(value, str):
        if value.isdigit():
            return int(value)
        else:
            try:
                return float(value)
            except Exception:
                return None
    return None


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


def is_binary(filePath):
    '''
        Returns True if the file is a binary file, False if it is a text file.
        Raises an Exception if the file could not be found.
        Source: https://stackoverflow.com/questions/898669/how-can-i-detect-if-a-file-is-binary-non-text-in-python
    '''
    textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
    with open(filePath, 'rb') as f:
        return bool(f.read(1024).translate(None, textchars))


def slugify(value, allow_unicode=False):
    '''
        Taken from https://github.com/django/django/blob/master/django/utils/text.py
        Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
        dashes to single dashes. Remove characters that aren't alphanumerics,
        underscores, or hyphens. Convert to lowercase. Also strip leading and
        trailing whitespace, dashes, and underscores.
    '''
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')



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



def get_library_available(libName, checkImport=False):
    '''
        Checks whether a Python library is available and returns a bool
        accordingly. Library names can be dot-separated as common in
        Python imports. If "checkImport" is True, the library is attempt-
        ed to be actually imported; if this fails, False is returned.
    '''
    try:
        if sys.version_info[1] <= 3:
            if importlib.find_loader(libName) is None:
                raise Exception('')
        else:
            if importlib.util.find_spec(libName) is None:
                raise Exception('')
        
        if checkImport:
            importlib.import_module(libName)
        
        return True
    except Exception:
        return False



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
    elif isinstance(boolean, int):
        return bool(boolean)
    elif isinstance(boolean, str):
        boolean = boolean.lower()
        if boolean.startswith('t') or boolean == '1' or boolean.startswith('y'):
            return True
        else:
            return False


def parse_parameters(data, params, absent_ok=True, escape=True, none_ok=True):
    '''
        Accepts a dict (data) and list (params) and assembles
        an output list of the entries in data under keys of params, in order of
        the latter.
        If params is a list of lists, the first entry of the nested list is the
        key, and the second the data type to which the value will be cast.
        Raises an Exception if typecasting fails.
        If absent_ok is True, missing keys will be skipped.
        If escape is True, sensitive characters will be escaped from strings.
        If none_ok is True, values may be None instead of the given data type.

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
        if not none_ok and value is not None:
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
        except Exception:
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
    except Exception:
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



def isAItask(taskName):
    '''
        Returns True if the taskName (str) is part of the AI
        task chain, or False if not (e.g., another type of task).
    '''
    tn = str(taskName).lower()
    return tn.startswith('aiworker') or \
        tn in ('aicontroller.get_training_images', 'aicontroller.get_inference_images')



def listDirectory(baseDir, recursive=False, images_only=True):
    '''
        Similar to glob's recursive file listing, but
        implemented so that circular softlinks are avoided.
        Removes the baseDir part (with trailing separator)
        from the files returned.
    '''
    if not images_only and not len(drivers.VALID_IMAGE_EXTENSIONS):
        drivers.init_drivers(False)      # should not be required
    files_disk = set()
    if not baseDir.endswith(os.sep):
        baseDir += os.sep
    def _scan_recursively(imgs, baseDir, fileDir, recursive):
        files = os.listdir(fileDir)
        for f in files:
            path = os.path.join(fileDir, f)
            if os.path.isfile(path) and (not images_only or os.path.splitext(f)[1].lower() in drivers.VALID_IMAGE_EXTENSIONS):
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



def fileListToHierarchy(fileList):
    '''
        Receives an Iterable of file names and converts it into a nested dict of
        dicts, corresponding to the folder hierarchy.
    '''
    assert isinstance(fileList, Iterable), 'Provided input is not an iterable list of files.'

    def _embed_file(tree, tokens, isDir):
        if not isinstance(tree, dict):
            return
        if isinstance(tokens, str):
            if isDir:
                tree[tokens] = {}
            else:
                tree[tokens] = None
        elif len(tokens) == 1:
            if isDir:
                tree[tokens[0]] = {}
            else:
                tree[tokens[0]] = None
        else:
            if tokens[0] not in tree:
                tree[tokens[0]] = {}
            _embed_file(tree[tokens[0]], tokens[1:], isDir)

    hierarchy = {}
    for file in fileList:
        tokens = file.split(os.sep)
        _embed_file(hierarchy, tokens, os.path.isdir(file))
    return hierarchy



def fileHierarchyToList(hierarchy):
    '''
        Receives a dict of dicts of files and returns a flattened list of files
        accordingly.
    '''
    assert isinstance(hierarchy, dict), 'Provided input is not a hierarchy of files.'

    fileList = []
    def _flatten_tree(tree, dirBase):
        if isinstance(tree, dict):
            for key in tree.keys():
                _flatten_tree(tree[key], os.path.join(dirBase, key))
        else:
            fileList.append(dirBase)
    _flatten_tree(hierarchy, '')
    return fileList



def rgbToHex(rgb):
    '''
        Receives a tuple/list with three int values and returns an
        HTML/CSS-compliant hex color string in format:
            "#RRGGBB"
    '''
    def clamp(x): 
        return max(0, min(x, 255))

    return '#{0:02x}{1:02x}{2:02x}'.format(clamp(rgb[0]), clamp(rgb[1]), clamp(rgb[2]))



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



def offsetColor(color, excluded=set(), distance=0):
    '''
        Receives a "color" (hex string) and a set of "excluded" color hex
        strings. Adjusts the color value up or down until its sum of absolute
        differences of RGB uint8 color values is larger than "distance" to the
        closest color in "excluded". Returns the modified color value as hex
        string accordingly.

        Used to adjust colors for segmentation projects (required due to
        anti-aliasing effects of HTML canvas).

        #TODO: offsetting colors slightly could bring them too close to the next one
        again...
    '''
    if not len(excluded):
        return color

    color_rgb = np.array(hexToRGB(color))
    excluded_rgb = set()
    for ex in excluded:
        excluded_rgb.add(hexToRGB(ex))
    excluded_rgb = np.array(list(excluded_rgb))

    dists = np.sum(np.abs(excluded_rgb - color_rgb), 1)
    if np.all(dists > distance):
        # color is already sufficiently spaced apart
        return color
    else:
        # need to offset; get closest
        closest = np.argmin(dists)
        componentDiffs = excluded_rgb[closest,:] - color_rgb

        # adjust in direction of difference
        diffPos = np.where(componentDiffs != 0)[0]
        if not len(diffPos):
            diffPos = np.array([0,1,2])

        color_rgb[diffPos] += (np.sign(componentDiffs[diffPos]) * componentDiffs[diffPos] * max(1, distance/len(diffPos))).astype(color_rgb.dtype)
        return rgbToHex(color_rgb.astype(np.uint8).tolist())



def randomHexColor(excluded=set(), distance=0):
    '''
        Creates a random HTML/CSS-compliant hex color string that is not already
        in the optional set/dict/list/tuple of "excluded" colors.

        If "distance" is an int or float, colors must be spaced apart by more
        than this value in terms of absolute summed numerical RGB differences.
    '''
    # create unique random color
    randomColor = '#{:06x}'.format(random.randint(10, 0xFFFFF0))
    
    #offset if needed
    if distance > 0 and len(excluded):
        randomColor = offsetColor(randomColor, excluded, distance)
    return randomColor



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

    except Exception:
        # something failed; set "corrupt" flag to False for image
        setImageCorrupt(dbConnector, project, imageID, True)
    
    finally:
        return img



def download_file(url, local_filename=None):
    '''
        Source: https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    '''
    if local_filename is None:
        local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return local_filename



FILENAMES_PROHIBITED_CHARS = (
    '&lt;',
    '<',
    '>',
    '&gt;',
    '..',
    '/',
    '\\',
    '|',
    '?',
    '*',
    ':'    # for macOS
)


DEFAULT_BAND_CONFIG = (
    'Red',
    'Green',
    'Blue'
)

DEFAULT_RENDER_CONFIG = {
    "bands": {
        "indices": {
            "red": 0,
            "green": 1,
            "blue": 2
        }
    },
    "grayscale": False,
    "white_on_black": False,
    "contrast": {
        "percentile": {
            "min": 0.0,
            "max": 100.0
        }
    },
    "brightness": 0
}