'''
    WMS service renderer.

    2023 Benjamin Kellenberger
'''

import re

from modules.Database.app import Database
from util.configDef import Config
from util import geospatial

from .abstract_renderer import AbstractRenderer
from .._functional import map_operations


class WMSRenderer(AbstractRenderer):
    '''
        Implementation of Web Map Service (WMS) server.
    '''

    SERVICE_NAME = 'wms'

    SERVICE_VERSIONS = ('1.3.0',)

    DEFAULT_SERVICE_VERSION = '1.3.0'

    SERVICE_TEMPLATES = (
        'get_capabilities',
        'layer_group',
        'layer'
    )

    def __init__(self, config: Config, db_connector: Database) -> None:
        super().__init__(config, db_connector)

        self.static_dir = self.config.getProperty('FileServer', 'staticfiles_dir')
        self.mime_pattern = re.compile(r'.*\/')


    def get_capabilities(self, projects: dict, base_url: str, request_params: dict) -> str:
        '''
            WMS GetCapabilities implementation.
        '''

        version = self.parse_version(request_params, True)
        if version is None:
            return self.render_error_template(10000,
                                                self.DEFAULT_SERVICE_VERSION,
                                                'Missing service version'), \
                    self.DEFAULT_RESPONSE_HEADERS

        projects_xml = ''
        for project, project_meta in projects.items():
            #TODO: add to project metadata
            # srid, extent = project_meta['srid'], project_meta['extent']
            srid = geospatial.get_project_srid(self.db_connector, project)
            extent = geospatial.get_project_extent(self.db_connector, project)

            #TODO: pre-filter
            if srid is None or extent is None:
                # no geodata in project
                continue

            base_args = {
                'srid': srid,
                'bbox_west': extent[0],
                'bbox_south': extent[1],
                'bbox_east': extent[2],
                'bbox_north': extent[3]
            }

            project_layers = ''

            # project images (always available)
            layer_args = base_args.copy()
            layer_args.update({
                'name': f'{project}{self.ID_DELIM}images',
                'title': 'images',
                'abstract': f'Image WMS for AIDE project {project}'
            })
            project_layers += self.render_service_template(version,
                                                            'layer',
                                                            layer_args,
                                                            False)

            if project_meta['annotation_type'] == 'segmentationmasks':
                # segmentation masks; encompass individual user's layers in a group
                user_details = ''
                for user in project_meta['users']:  #TODO
                    layer_args.update({
                        'name': f'{project}{self.ID_DELIM}annotation{self.ID_DELIM}{user}',
                        'title': user,
                        'abstract': f'AIDE project {project}: annotations by user {user}',
                    })
                    user_details += self.render_service_template(version,
                                                                'layer',
                                                                layer_args,
                                                                False)
                group_args = base_args.copy()
                group_args.update({
                    'title': 'annotations',
                    'name': f'{project}{self.ID_DELIM}annotation',
                    'abstract': f'AIDE project {project}: segmentation annotations per user',
                    'layers': user_details
                })
                project_layers += self.render_service_template(version,
                                                                'layer_group',
                                                                group_args,
                                                                False)

            #TODO: predictions

            # combine all layers into a project-wide group
            project_args = base_args.copy()
            project_args.update({
                'title': project,
                'name': project,
                'abstract': f'WMS for AIDE project {project}',
                'layers': project_layers
            })
            projects_xml += self.render_service_template(version,
                                                            'layer_group',
                                                            project_args,
                                                            False)

        # combine all projects
        capabilities_args = self.DEFAULT_CAPABILITIES_ARGS.copy()
        capabilities_args.update({
            'version': version,
            'online_resource_href': base_url,
            'base_href': base_url,
            'project_meta': projects_xml
        })
        return self.render_service_template(version,
                                            'get_capabilities',
                                            capabilities_args,
                                            True), \
                    self.DEFAULT_RESPONSE_HEADERS


    def get_map(self, projects: dict, base_url: str, request_params: dict) -> object:
        '''
            WMS GetMap implementation.
        '''
        version = self.parse_version(request_params, False)
        layer = request_params.get('LAYER', request_params.get('LAYERS', None))
        project, layer_name, entity = self._decode_layer_name(layer)
        if project not in projects:
            # invalid/inaccessible project requested
            return self.render_error_template(11000,
                                                version,
                                                f'Invalid layer "{layer}"'), \
                        self.DEFAULT_RESPONSE_HEADERS
        project_meta = projects[project]
        srid = project_meta['srid']
        bbox = request_params.get('BBOX', None)
        if bbox is not None:
            if request_params.get('FLIP_COORDINATES', False):
                bbox = (
                    bbox[1], bbox[0],
                    bbox[3], bbox[2]
                )
        width, height = request_params.get('WIDTH', None), request_params.get('HEIGHT', None)
        if all(item is not None for item in (bbox, width, height)):
            resolution = (
                (bbox[2]-bbox[0]) / float(width),
                (bbox[3]-bbox[1]) / float(height)
            )
        else:
            resolution = None

        mime_type = request_params.get('FORMAT', 'image/png')
        image_ext = re.sub(
            self.mime_pattern,
            'c:/fakepath/tile.',
            mime_type
        )

        response_headers = self.DEFAULT_RESPONSE_HEADERS.copy()

        if layer_name == 'images':
            bytes_obj = map_operations.get_map_images(self.db_connector,
                                                        self.static_dir,
                                                        project,
                                                        project_meta,
                                                        bbox,
                                                        srid,
                                                        resolution,
                                                        image_ext,
                                                        raw=False)
            response_headers.update({
                'Content-Type': mime_type,
                'Content-Length': len(bytes_obj)
            })
            return bytes_obj, response_headers

        if layer_name in ('annotation', 'prediction'):
            if project_meta[f'{layer_name}_type'] != 'segmentationmasks':
                # requesting raster annotations/predictions from vector project
                return self.render_error_template(11001,
                                                version,
                                                'Unsupported operation'), \
                        self.DEFAULT_RESPONSE_HEADERS
            bytes_obj = map_operations.get_map_segmentation(self.db_connector,
                                                            self.static_dir,
                                                            project,
                                                            project_meta,
                                                            layer_name,
                                                            entity,
                                                            bbox,
                                                            srid,
                                                            resolution,
                                                            image_ext,
                                                            raw=False)
            response_headers.update({
                'Content-Type': mime_type,
                'Content-Length': len(bytes_obj)
            })
            return bytes_obj, response_headers

        return self.render_error_template(11002,
                                            version,
                                            f'Invalid layer name "{layer}"'), \
                self.DEFAULT_RESPONSE_HEADERS


    def _load_service_requests(self):
        self._register_service_request('GetCapabilities', self.get_capabilities)
        self._register_service_request('GetMap', self.get_map)
