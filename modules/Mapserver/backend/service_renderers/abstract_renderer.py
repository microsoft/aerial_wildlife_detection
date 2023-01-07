'''
    Abstract service renderer.

    2023 Benjamin Kellenberger
'''

import os
from typing import Tuple, Iterable, Callable
from abc import abstractmethod
from collections import defaultdict

from modules.Database.app import Database
from util.configDef import Config


class AbstractRenderer:
    '''
        Abstract base class for Web geodata service renderers.
    '''

    SERVICE_NAME = ''

    SERVICE_VERSIONS = ()

    DEFAULT_SERVICE_VERSION = ''

    SERVICE_TEMPLATES = (
        'get_capabilities',
    )

    DEFAULT_CAPABILITIES_ARGS = {
        'version': '',
        'name': 'AIDE_Mapserver',
        'title': 'AIDE Mapserver',
        'abstract': 'AIDE Mapserver',
        'keywords': '',
        'online_resource_href': '',
        'base_href': '',
        'project_meta': '',

        'contact_name': '',
        'contact_organization': '',
        'contact_address': '',
        'contact_city': '',
        'contact_state_province': '',
        'contact_postcode': '',
        'contact_country': '',
        'contact_voice': '',
        'contact_fax': '',
        'contact_email': ''
    }

    ID_DELIM = '_'

    DEFAULT_RESPONSE_HEADERS = {
        'Content-Type': 'text/xml',

        #TODO: CORS headers
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Credentials': 'true',
        'Access-Control-Allow-Methods': 'GET, POST',
        'Access-Control-Allow-Headers': '*'
    }

    def __init__(self, config: Config, db_connector: Database) -> None:
        self.config = config
        self.db_connector = db_connector

        self.service_templates = defaultdict(dict)
        self._load_service_templates()

        self.service_requests = {}
        self._load_service_requests()


    def _load_service_templates(self) -> None:
        for version in self.SERVICE_VERSIONS:
            for template_name in self.SERVICE_TEMPLATES:
                with open(os.path.join(
                    'modules/Mapserver/static/xml',
                    self.SERVICE_NAME,
                    version,
                    f'{template_name}.xml'
                ), 'r', encoding='utf-8') as f_template:
                    template = f_template.read()
                self.service_templates[version][template_name] = template
        # error template
        with open('modules/Mapserver/static/xml/error.xml', 'r', encoding='utf-8') as f_error:
            template = f_error.read()
        self.service_templates['error'] = template


    def parse_version(self, request_params: dict, default: bool=False) -> str:
        '''
            Receives one or more service version(s) possibly present in request parameters and
            returns a single one, compatible with the current service. If "return_default" is True,
            the default service version is returned (else None).
        '''
        versions = []
        for key in ('VERSION', 'ACCEPTVERSIONS'):
            if key in request_params:
                versions = request_params[key].strip().split(',')
                break
        for version in versions:
            version = version.strip()
            if version in self.SERVICE_VERSIONS:
                return version
        if default:
            return self.DEFAULT_SERVICE_VERSION
        return None


    @classmethod
    def _encode_layer_name(cls, project: str, layer: str=None, entity: str=None) -> str:
        escape = f'{cls.ID_DELIM}{cls.ID_DELIM}'
        layer_name = project.replace(cls.ID_DELIM, escape)
        if layer is not None:
            layer_name += f'{cls.ID_DELIM}{layer.replace(cls.ID_DELIM, escape)}'
        if entity is not None:
            layer_name += f'{cls.ID_DELIM}{entity.replace(cls.ID_DELIM, escape)}'
        return layer_name


    @classmethod
    def _decode_layer_name(cls, layer_name: str) -> tuple:
        layer_name = layer_name.strip().replace(f'{cls.ID_DELIM}{cls.ID_DELIM}', cls.ID_DELIM)
        tokens = layer_name.split(cls.ID_DELIM)
        while len(tokens) < 3:
            # no layer name (and username) specified
            tokens.append(None)
        return tokens


    def render_error_template(self,
                                version: Iterable,
                                error_code: str,
                                message: str) -> str:
        '''
            TODO
        '''
        return self.service_templates['error'].format(
            version=version,
            code=str(error_code),
            message=message
        )


    def render_service_template(self,
                                version: str,
                                request_type: str,
                                format_args: dict,
                                render_error: bool=False) -> str:
        '''
            TODO
        '''
        def _error(code, service_version, message):
            if render_error:
                return self.service_templates['error'].format(
                    version=service_version,
                    code=code,
                    message=message
                )
            return ''
        if version is None:
            return _error(10000, self.DEFAULT_SERVICE_VERSION, 'Missing service version')

        if version not in self.service_templates:
            return _error(10001, self.DEFAULT_SERVICE_VERSION,
                            f'Unsupported service version {version}')

        service_group = self.service_templates[version]
        if request_type not in service_group:
            return _error(10002, version, f'Unsupported request type {request_type}')

        template = service_group[request_type]
        try:
            return template.format_map(format_args)
        except Exception as exc:
            return _error(10003, version, f'Internal error (message: "{exc}")')


    def _register_service_request(self, request_name: str, function: Callable) -> None:
        '''
            Registration of Mapserver request functionality (e.g., "GetCapabilities") with
            callables.
        '''
        self.service_requests[request_name.lower()] = function


    @abstractmethod
    def _load_service_requests(self):
        raise NotImplementedError('Not implemented for abstract base class')


    def service(self, request_name: str,
                    projects: dict,
                    base_url: str,
                    request_params: dict) -> Tuple[object, dict]:
        '''
            Calls the renderer's service by a given request_name (case-insensitive) and arguments.
            Raises an Exception if the request has not been found.
        '''
        req_name = request_name.lower()
        if req_name not in self.service_requests:
            raise Exception(f'Invalid request name "{request_name}"')
        return self.service_requests[req_name](projects, base_url, request_params)


    def __call__(self, request_name: str,
                    projects: dict,
                    base_url: str,
                    request_params: dict) -> Tuple[object, dict]:
        return self.service(request_name, projects, base_url, request_params)
