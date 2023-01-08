'''
    Serves files, such as images, from a local directory.

    2019-23 Benjamin Kellenberger
'''

import os
from io import BytesIO
from bottle import static_file, request, abort, _file_iter_range, parse_range_header, HTTPResponse
from util.cors import enable_cors
from util import helpers
from util.drivers import is_web_compatible, GDALImageDriver        #TODO


class FileServer():

    DEFAULT_IMAGE_KWARGS = {
        'driver': 'GTiff',
        'compress': 'lzw'
    }
    DEFAULT_IMAGE_MIME_TYPE = 'image/tiff'

    def __init__(self, config, app, dbConnector, verbose_start=False):
        self.config = config
        self.app = app

        if verbose_start:
            print('FileServer'.ljust(helpers.LogDecorator.get_ljust_offset()), end='')

        if not helpers.is_fileServer(config):
            if verbose_start:
                helpers.LogDecorator.print_status('fail')
            raise Exception('Not a valid FileServer instance.')

        self.login_check = None
        try:
            self.static_dir = self.config.getProperty('FileServer', 'staticfiles_dir')
            self.static_address_suffix = self.config.getProperty('FileServer',
                                        'staticfiles_uri_addendum', type=str, fallback='').strip()

            assert GDALImageDriver.init_is_available()  #TODO

            self._initBottle()
        except Exception as exc:
            if verbose_start:
                helpers.LogDecorator.print_status('fail')
            raise Exception(f'Could not launch FileServer (message: "{str(exc)}").') from exc

        if verbose_start:
            helpers.LogDecorator.print_status('ok')


    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun


    def _initBottle(self):

        ''' static routing to files '''
        @enable_cors
        @self.app.route(os.path.join('/', self.static_address_suffix, '/<project>/files/<path:path>'))
        def send_file(project, path):
            file_path = os.path.join(self.static_dir, project, path)
            need_conversion = not is_web_compatible(file_path)
            window = request.params.get('window', None)
            bands = request.params.get('bands', None)
            if need_conversion or window is not None or bands is not None:
                # load from disk and crop
                driver_kwargs = {}
                if isinstance(window, str):
                    window = [int(w) for w in window.strip().split(',')]
                    driver_kwargs['window'] = window
                if isinstance(bands, str):
                    bands = [int(band) for band in bands.strip().split(',')]
                    driver_kwargs['bands'] = bands

                # check if file needs conversion
                if need_conversion:
                    driver_kwargs.update(self.DEFAULT_IMAGE_KWARGS)

                bytes_arr = GDALImageDriver.disk_to_bytes(file_path, **driver_kwargs)
                clen = len(bytes_arr)

                headers = {}
                # headers['Content-type'] = mime_type
                headers['Content-Disposition'] = f'attachment; filename="{path}"'

                ranges = request.environ.get('HTTP_RANGE')
                if 'HTTP_RANGE' in request.environ:
                    # need to send bytes in chunks
                    fhandle = BytesIO(bytes_arr)
                    ranges = list(parse_range_header(request.environ['HTTP_RANGE'], clen))
                    offset, end = ranges[0]
                    headers['Content-Range'] = f'bytes {offset}-{end-1}/{clen}'
                    headers['Content-Length'] = str(end-offset)
                    fhandle = _file_iter_range(fhandle, offset, end-offset)
                    return HTTPResponse(fhandle, status=206, **headers)

                return HTTPResponse(bytes_arr, status=200, **headers)

            # full image; return static file directly
            #TODO: only go this route if fully-supported image (i.e., parseable by client)
            return static_file(path, root=os.path.join(self.static_dir, project))


        @enable_cors
        @self.app.get('/getFileServerInfo')
        def get_file_server_info():
            '''
                Returns immutable parameters like the file directory
                and address suffix.
                User must be logged in to retrieve this information.
            '''
            if not self.loginCheck(extend_session=True):
                abort(401, 'forbidden')

            return {
                'staticfiles_dir': self.static_dir,
                'staticfiles_uri_addendum': self.static_address_suffix
            }
