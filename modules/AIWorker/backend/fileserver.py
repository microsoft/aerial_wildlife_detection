'''
    Private file server wrapper, to be used explicitly by the backend.
    Note: this instance does not do any user verification or authentication
    check; it is therefore imperative that it may never be exposed to the
    frontend.
    An instance of this FileServer class may be provided to the AIModel instead,
    and serves as a gateway to the project's actual file server.

    2019-22 Benjamin Kellenberger
'''

import os
from urllib import request
import urllib.parse
from urllib.error import HTTPError
from util.helpers import is_localhost
from util import drivers
drivers.init_drivers()


class FileServer:

    def __init__(self, config):
        self.config = config

        # check if file server runs on the same machine
        self.isLocal = self._check_running_local()

        # base URI
        if self.isLocal:
            # URI is a direct file path
            self.baseURI = self.config.getProperty('FileServer', 'staticfiles_dir')
        
        else:
            self.baseURI = self.config.getProperty('Server', 'dataServer_uri')
            

    
    def _check_running_local(self):
        '''
            For simpler projects one might run both the AIWorker(s) and the FileServer
            module on the same machine. In this case we don't route file requests through
            the (loopback) network, but load files directly from disk. This is the case if
            the configuration's 'dataServer_uri' item specifies a local address, which we
            check for here.
        '''
        baseURI = self.config.getProperty('Server', 'dataServer_uri')
        return is_localhost(baseURI)

    

    def getFile(self, project, filename):
        '''
            Returns the file as a byte array.
            If FileServer module runs on same instance as AIWorker,
            the file is directly loaded from the local disk.
            Otherwise an HTTP request is being sent.
        '''
        try:
            #TODO: make generator that yields bytes?
            if not self.isLocal:
                filename = urllib.parse.quote(filename)
            localSpec = ('files' if not self.isLocal else '')
            if project is not None:
                queryPath = os.path.join(self.baseURI, project, localSpec, filename)
            else:
                queryPath = os.path.join(self.baseURI, filename)
            
            if '..' in queryPath or filename.startswith(os.sep):
                # parent and absolute paths are not allowed (to protect the system and other projects)
                raise Exception('Parent accessors ("..") and absolute paths ("{}path") are not allowed.'.format(os.sep))

            if self.isLocal:
                # load file from disk
                driver = drivers.get_driver(queryPath)      #TODO: try-except?
                bytea = driver.disk_to_bytes(queryPath)
                # with open(queryPath, 'rb') as f:
                #     bytea = f.read()
            else:
                response = request.urlopen(queryPath)
                bytea = response.read()

        except HTTPError as httpErr:
            print('HTTP error')
            print(httpErr)
            bytea = None

        except Exception as err:
            print(err)  #TODO: don't throw an exception, but let worker handle it?
            bytea = None

        return bytea
    


    def getImage(self, project, filename):
        '''
            Returns an image as a NumPy ndarray.
            If FileServer module runs on same instance as AIWorker,
            the file is directly loaded from the local disk.
            Otherwise an HTTP request is being sent.
        '''
        img = None
        try:
            if not self.isLocal:
                filename = urllib.parse.quote(filename)
            localSpec = ('files' if not self.isLocal else '')
            if project is not None:
                queryPath = os.path.join(self.baseURI, project, localSpec, filename)
            else:
                queryPath = os.path.join(self.baseURI, filename)
            
            if '..' in queryPath or filename.startswith(os.sep):
                # parent and absolute paths are not allowed (to protect the system and other projects)
                raise Exception('Parent accessors ("..") and absolute paths ("{}path") are not allowed.'.format(os.sep))

            if self.isLocal:
                # load file from disk
                qpath_stripped, window = drivers.strip_window(queryPath)
                driver = drivers.get_driver(qpath_stripped)      #TODO: try-except?
                img = driver.load_from_disk(qpath_stripped, window=window)
            else:
                response = request.urlopen(queryPath)
                bytea = response.read()
                img = drivers.load_from_bytes(bytea)

        except HTTPError as httpErr:
            print('HTTP error')
            print(httpErr)
        except Exception as err:
            print(err)  #TODO: don't throw an exception, but let worker handle it?

        return img



    def putFile(self, project, bytea, filename):
        '''
            Saves a file to disk.
            TODO: requires locally running FileServer instance for now.
        '''
        #TODO: What about remote file server? Might need to do authentication and sanity checks...
        if project is not None:
            path = os.path.join(self.baseURI, project, filename)
        else:
            path = os.path.join(self.baseURI, filename)

        if '..' in path or filename.startswith(os.sep):
            # parent and absolute paths are not allowed (to protect the system and other projects)
            raise Exception('Parent accessors ("..") and absolute paths ("{}path") are not allowed.'.format(os.sep))

        with open(path, 'wb') as f:
            f.write(bytea)
        print('Wrote file to disk: ' + filename)    #TODO

    

    def get_secure_instance(self, project):
        '''
            Returns a wrapper class to the "getFile" and "putFile"
            functions that disallow access to other projects
            than the one included.
        '''
        this = self
        class _secure_file_server:
            def getFile(self, filename):
                return this.getFile(project, filename)
            def getImage(self, filename):
                return this.getImage(project, filename)
            def putFile(self, bytea, filename):
                return this.putFile(project, bytea, filename)
        
        return _secure_file_server()