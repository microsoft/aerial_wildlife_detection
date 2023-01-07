'''
    WCS service renderer.

    2023 Benjamin Kellenberger
'''

from typing import Tuple
import re
import numpy as np

from modules.Database.app import Database
from util.configDef import Config
from util import geospatial

from .abstract_renderer import AbstractRenderer
from .._functional import map_operations


class WCSRenderer(AbstractRenderer):
    '''
        Implementation of Web Coverage Service (WCS) server.
    '''

    SERVICE_NAME = 'wcs'

    SERVICE_VERSIONS = ('1.1.0',)

    DEFAULT_SERVICE_VERSION = '1.1.0'

    SERVICE_TEMPLATES = (
        'get_capabilities',
        'coverage_summary',
        'describe_coverage',
        'field'
    )

    def __init__(self, config: Config, db_connector: Database) -> None:
        super().__init__(config, db_connector)

        self.static_dir = self.config.getProperty('FileServer', 'staticfiles_dir')
        self.mime_pattern = re.compile(r'.*\/')


    def get_capabilities(self, projects: dict,
                                base_url: str,
                                request_params: dict) -> Tuple[str, dict]:
        '''
            WCS GetCapabilities implementation.
        '''
        version = self.parse_version(request_params, False)
        if version is None:
            return self.render_error_template(10000,
                                                self.DEFAULT_SERVICE_VERSION,
                                                'Missing service version'), \
                    self.DEFAULT_RESPONSE_HEADERS

        projects_xml = ''
        layer_identifiers = ''
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
            layer_id = self._encode_layer_name(project, 'images')
            layer_args = base_args.copy()
            layer_args.update({
                'identifier': f'<Identifier>{layer_id}</Identifier>',
                'title': 'images',
                'abstract': f'Image WCS for AIDE project {project}',
                'children': ''
            })
            layer_identifiers += f'<ows:Value>{layer_id}</ows:Value>'
            project_layers += self.render_service_template(version,
                                                            'coverage_summary',
                                                            layer_args,
                                                            False)

            if project_meta['annotation_type'] == 'segmentationmasks':
                # segmentation masks; encompass individual user's layers in a group
                user_details = ''
                for user in project_meta['users']:
                    layer_name = self._encode_layer_name(project, 'annotation', user)
                    layer_args.update({
                        'identifier': f'<Identifier>{layer_name}</Identifier>',
                        'title': user,
                        'abstract': f'AIDE project {project}: annotations by user {user}',
                        'children': ''
                    })
                    layer_identifiers += f'<ows:Value>{layer_name}</ows:Value>'
                    user_details += self.render_service_template(version,
                                                                'coverage_summary',
                                                                layer_args,
                                                                False)
                group_args = base_args.copy()
                group_args.update({
                    'identifier': '',
                    'title': 'annotations',
                    'abstract': f'AIDE project {project}: segmentation annotations per user',
                    'children': user_details
                })
                project_layers += self.render_service_template(version,
                                                                'coverage_summary',
                                                                group_args,
                                                                False)

            #TODO: predictions

            # combine all layers into a project-wide group
            project_args = base_args.copy()
            project_args.update({
                'identifier': '',
                'title': project,
                'abstract': f'WCS for AIDE project {project}',
                'children': project_layers
            })
            projects_xml += self.render_service_template(version,
                                                            'coverage_summary',
                                                            project_args,
                                                            False)

        # combine all projects
        capabilities_args = self.DEFAULT_CAPABILITIES_ARGS.copy()
        capabilities_args.update({
            'version': version,
            'online_resource_href': base_url,
            'base_href': base_url,
            'project_meta': projects_xml,
            'identifiers': layer_identifiers
        })
        return self.render_service_template(version,
                                            'get_capabilities',
                                            capabilities_args,
                                            True), \
                self.DEFAULT_RESPONSE_HEADERS


    def describe_coverage(self, projects: dict,
                                base_url: str,
                                request_params: dict) -> Tuple[str, dict]:
        '''
            WCS DescribeCoverage implementation.
        '''
        version = self.parse_version(request_params, False)
        identifier = request_params.get('IDENTIFIER', request_params.get('IDENTIFIERS'))
        project, layer_name, entity = self._decode_layer_name(identifier)
        if project not in projects:
            # invalid/inaccessible project requested
            return self.render_error_template(11000,
                                                version,
                                                f'Invalid identifier "{identifier}"'), \
                    self.DEFAULT_RESPONSE_HEADERS
        project_meta = projects[project]
        srid, extent = project_meta['srid'], project_meta['extent']

        #TODO: support multiple projects?

        # assemble available fields for project
        fields = ''
        field_args = {}
        if layer_name in ('images', None):
            # images
            field_args = {
                'title': 'images',
                'identifier': self._encode_layer_name(project, 'images'),
                'abstract': f'AIDE project {project}: images',
                'band_keys': '\n'.join([
                    f'<Key>{band}</Key>'
                    for band in project_meta['band_config']
                ])
            }
            fields += self.render_service_template(version, 'field', field_args, False)

        # annotations
        if layer_name in ('annotation', None) and \
            project_meta['annotation_type'] == 'segmentationmasks':
            for user in project_meta['users']:  #TODO
                field_args.update({
                    'title': user,
                    'identifier': self._encode_layer_name(project, 'annotation', user),
                    'abstract': f'AIDE project {project}: annotations by user {user}',
                    'band_keys': '<Key>0</Key>'
                })
                fields += self.render_service_template(version, 'field', field_args, False)

        # predictions
        if layer_name in ('prediction', None) and \
            project_meta['prediction_type'] == 'segmentationmasks':
            pass
            #TODO: for each model state
            # for user in project_meta['users']:  #TODO
            #     field_args.update({
            #         'title': user,
            #         'identifier': TODO:encode,
            #         'abstract': f'AIDE project {project}: annotations by user {user}',
            #         'band_keys': '<Key>0</Key>'
            #     })
            #     fields += self.render_service_template(version, 'field', field_args, False)

        # assemble all together
        format_args = {
            'title': identifier,
            'abstract': 'TODO',
            'identifier': identifier,
            'project_name': project,
            'bbox_west': extent[0],
            'bbox_south': extent[1],
            'bbox_east': extent[2],
            'bbox_north': extent[3],
            'crs': f'EPSG:{srid}',
            'srid': srid,
            'fields': fields
        }
        return self.render_service_template(version, 'describe_coverage', format_args, True), \
                self.DEFAULT_RESPONSE_HEADERS


    def get_coverage(self, projects: dict,
                            base_url: str,
                            request_params: dict) -> Tuple[object, dict]:
        '''
            WCS GetCoverage implementation.
        '''
        version = self.parse_version(request_params, False)
        identifier = request_params.get('IDENTIFIER', request_params.get('IDENTIFIERS'))
        project, layer_name, entity = self._decode_layer_name(identifier)
        if project not in projects:
            # invalid/inaccessible project requested
            return self.render_error_template(11000,
                                                version,
                                                f'Invalid identifier "{identifier}"'), \
                    self.DEFAULT_RESPONSE_HEADERS
        project_meta = projects[project]
        srid = project_meta['srid']
        bbox = request_params.get('BBOX', None)
        flip_coordinates = request_params.get('FLIP_COORDINATES', False)

        resolution = None
        grid_offsets = request_params.get('GRIDOFFSETS', None)
        if isinstance(grid_offsets, str) and len(grid_offsets) > 0:
            grid_offsets = grid_offsets.strip().split(',')
            if len(grid_offsets) >= 2:
                try:
                    resolution = (
                        np.abs(float(grid_offsets[0])),
                        np.abs(float(grid_offsets[1]))
                    )
                    if flip_coordinates:
                        resolution = (resolution[1], resolution[0])
                except Exception:
                    resolution = None

        if bbox is not None:
            if flip_coordinates:
                bbox = (
                    bbox[1], bbox[0],
                    bbox[3], bbox[2]
                )
            if resolution is None and all(dim in request_params for dim in ('WIDTH', 'HEIGHT')):
                width, height = request_params['WIDTH'], request_params['HEIGHT']
                resolution = (
                    (bbox[2]-bbox[0]) / float(width),
                    (bbox[3]-bbox[1]) / float(height)
                )

        mime_type = request_params.get('FORMAT', 'image/tiff')
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
                                                        raw=True)
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
                                                            raw=True)
            response_headers.update({
                'Content-Type': mime_type,
                'Content-Length': len(bytes_obj)
            })
            return bytes_obj, response_headers

        return self.render_error_template(11002,
                                            version,
                                            f'Invalid identifier name "{identifier}"'), \
                        self.DEFAULT_RESPONSE_HEADERS


    def _load_service_requests(self):
        self._register_service_request('GetCapabilities', self.get_capabilities)
        self._register_service_request('DescribeCoverage', self.describe_coverage)
        self._register_service_request('GetCoverage', self.get_coverage)
