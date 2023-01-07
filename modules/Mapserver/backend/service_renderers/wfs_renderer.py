'''
    WFS service renderer.

    2023 Benjamin Kellenberger
'''

from typing import Tuple
import re
from urllib.parse import urlparse
from psycopg2 import sql
import numpy as np
import rasterio

from modules.Database.app import Database
from util.configDef import Config
from util import geospatial

from .abstract_renderer import AbstractRenderer
from .._functional.gml_formatter import GMLFormatter
from .. import METADATA_SPEC


class WFSRenderer(AbstractRenderer):
    '''
        Implementation of Web Feature Service (WFS) server.
    '''

    SERVICE_NAME = 'wfs'

    SERVICE_VERSIONS = ('2.0.0',)

    DEFAULT_SERVICE_VERSION = '2.0.0'

    SERVICE_TEMPLATES = (
        'get_capabilities',
        'describe_feature_type',
        'get_feature',
        'feature_type',
        'element'
    )

    def __init__(self, config: Config, db_connector: Database) -> None:
        super().__init__(config, db_connector)

        self.static_dir = self.config.getProperty('FileServer', 'staticfiles_dir')
        self.mime_pattern = re.compile(r'.*\/')

        self.gml_formatter = GMLFormatter()


    @staticmethod
    def _get_gml_geometry_property(project_meta: dict, anno_type: str) -> str:
        anno_type = project_meta[anno_type]
        if anno_type in ('labels', 'polygons', 'boundingboxes', 'image-outlines'):
            return 'gml:polygonProperty'
        if anno_type == 'points':
            return 'gml:PointPropertyType'
        # other: e.g., segmentation masks
        return None


    def get_capabilities(self, projects: dict,
                                base_url: str,
                                request_params: dict) -> Tuple[str, dict]:
        '''
            WFS GetCapabilities implementation.
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

            # project image outlines (always available)
            layer_args = base_args.copy()
            layer_args.update({
                'name': self._encode_layer_name(project, 'image-outlines'),
                'title': 'image outlines',
                'abstract': f'Image extent WFS for AIDE project {project}'
            })
            project_layers += self.render_service_template(version,
                                                            'feature_type',
                                                            layer_args,
                                                            False)

            if project_meta['annotation_type'] in \
                    ('labels', 'points', 'polygons', 'boundingboxes'):
                # vector annotations
                layer_args.update({
                    'name': self._encode_layer_name(project, 'annotation'),
                    'title': 'annotations',
                    'abstract': f'AIDE project {project}: annotations',
                })
                project_layers += self.render_service_template(version,
                                                            'feature_type',
                                                            layer_args,
                                                            False)
            if project_meta['prediction_type'] in \
                    ('labels', 'points', 'polygons', 'boundingboxes'):
                # vector predictions
                layer_args.update({
                    'name': self._encode_layer_name(project, 'prediction'),
                    'title': 'predictions',
                    'abstract': f'AIDE project {project}: predictions',
                })
                project_layers += self.render_service_template(version,
                                                            'feature_type',
                                                            layer_args,
                                                            False)

            #TODO: group
            projects_xml += project_layers

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


    def describe_feature_type(self, projects: dict,
                                    base_url: str,
                                    request_params: dict) -> Tuple[object, dict]:
        '''
            WFS DescribeFeatureType implementation.
        '''
        version = self.parse_version(request_params, False)
        type_name = request_params.get('TYPENAME', request_params.get('TYPENAMES', None))
        project, layer_name, entity = self._decode_layer_name(type_name)
        if project not in projects:
            # invalid/inaccessible project requested
            return self.render_error_template(11000,
                                                version,
                                                f'Invalid type name "{type_name}"'), \
                    self.DEFAULT_RESPONSE_HEADERS
        project_meta = projects[project]
        srid, extent = project_meta['srid'], project_meta['extent']

        # get entries
        elements = ''

        # image outlines
        if layer_name in ('image-outlines', None):
            meta_fields = ''
            for key, value_type in METADATA_SPEC['image-outlines'].items():
                meta_fields += f'''
                    <element name="{key}" minOccurs="1" maxOccurs="1"
                        nillable="true" type="{value_type}"/>
                '''
            feature_args = {
                'elem_name': self._encode_layer_name(project, 'image-outlines'),
                'gml_geometry_type': 'gml:polygonProperty',
                'meta_fields': meta_fields
            }
            elements += self.render_service_template(version, 'element', feature_args, False)

        # annotations
        geom_anno = self._get_gml_geometry_property(project_meta, 'annotation_type')
        if isinstance(geom_anno, str) and layer_name in ('annotation', None):
            meta_fields = ''
            for key, value_type in METADATA_SPEC['annotation'].items():
                meta_fields += f'''
                    <element name="{key}" minOccurs="1" maxOccurs="1"
                        nillable="true" type="{value_type}"/>
                '''
            feature_args = {
                'elem_name': self._encode_layer_name(project, 'annotation'),
                'gml_geometry_type': geom_anno,
                'meta_fields': meta_fields
            }
            elements += self.render_service_template(version, 'element', feature_args, False)

        # predictions
        geom_pred = self._get_gml_geometry_property(project_meta, 'prediction_type')
        if isinstance(geom_pred, str) and layer_name in ('prediction', None):
            meta_fields = ''
            for key, value_type in METADATA_SPEC['prediction'].items():
                meta_fields += f'''
                    <element name="{key}" minOccurs="1" maxOccurs="1"
                        nillable="true" type="{value_type}"/>
                '''
            feature_args = {
                'elem_name': self._encode_layer_name(project, 'prediction'),
                'gml_geometry_type': geom_pred,
                'meta_fields': meta_fields
            }
            elements += self.render_service_template(version, 'element', feature_args, False)

        # assemble formatting args
        format_args = {
            'project_name': project,
            'online_resource_href': base_url,
            'elements': elements,
            'meta_fields_image_outlines': ''
        }
        return self.render_service_template(version, 'describe_feature_type', format_args, True), \
                self.DEFAULT_RESPONSE_HEADERS


    def _encode_features(self, meta: dict,
                            relation_name: str,
                            type_name: str,
                            version: str,
                            srid: int,
                            flip_coordinates: bool=False) -> str:
        transform = rasterio.transform.Affine.from_gdal(*meta['affine_transform'])

        delim = ','
        if 'coordinates' in meta:
            # polygon
            coords = np.reshape(np.array(meta['coordinates']), (-1, 2)).T
            # coords = np.concatenate((coords, coords[:,0]), 1)
        elif all(key in meta for key in ('x', 'y')):
            if all(key in meta for key in ('width', 'height')):
                # bounding box; convert to polygon
                coords = np.array([
                    [meta['x']-meta['width']/2, meta['y']-meta['height']/2],
                    [meta['x']-meta['width']/2, meta['y']+meta['height']/2],
                    [meta['x']+meta['width']/2, meta['y']+meta['height']/2],
                    [meta['x']+meta['width']/2, meta['y']-meta['height']/2],
                    [meta['x']-meta['width']/2, meta['y']-meta['height']/2]
                ]).T
            else:
                # point
                delim = ' '
                coords = np.array([meta['x'], meta['y']])[:, np.newaxis]
        else:
            # image label or image outline; turn image envelope into polygon
            coords = np.array([
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0]
            ]).T

        # geocode coordinates
        coords *= np.array([meta['img_width'], meta['img_height']])[:,np.newaxis]
        lon, lat = transform * coords
        if flip_coordinates:
            coords = np.stack((lat, lon), 1).tolist()
        else:
            coords = np.stack((lon, lat), 1).tolist()
        coords = ' '.join(f'{coord[0]}{delim}{coord[1]}' for coord in coords)

        # translate to GML
        meta.update({
            'anno_id': str(meta['id']),
            'srid': srid,
            'coords': coords
        })
        try:
            return self.gml_formatter.to_gml(meta, relation_name, type_name, version)
        except Exception:
            return ''


    def get_feature(self, projects: dict, base_url: str, request_params: dict) -> Tuple[str, dict]:
        '''
            WFS GetFeature implementation.
        '''
        version = self.parse_version(request_params, False)
        type_names = request_params.get('TYPENAME', request_params.get('TYPENAMES', None))
        if isinstance(type_names, str):
            type_names = (type_names,)
        flip_coordinates = request_params.get('FLIP_COORDINATES', False)
        try:
            count = int(request_params.get('COUNT', -1))
        except Exception:
            count = -1
        count_str = ''
        if count > 0:
            count_str = 'LIMIT %s'

        base_url = urlparse(base_url)
        base_url = f'{base_url.scheme}://{base_url.netloc}'

        gml_features = ''
        for type_name in type_names:
            project, layer_name, entity = self._decode_layer_name(type_name)
            if project not in projects:
                # feature of invalid/inaccessible project requested
                continue

            project_meta = projects[project]

            srid = project_meta['srid']
            bbox = request_params.get('BBOX', None)
            usernames = project_meta['users']           #TODO: also check with entity

            # find all images and geometries
            query_args = []
            bbox_sql, bbox_gml = '', ''
            if bbox is not None:
                if flip_coordinates:
                    bbox = (
                        bbox[1], bbox[0],
                        bbox[3], bbox[2]
                    )
                query_args = [*bbox, srid]
                bbox_sql = '''
                    WHERE ST_Intersects(
                        img.extent,
                        ST_MakeEnvelope(
                                %s, %s, %s, %s,
                                %s
                        )
                    )
                '''
                bbox_gml = f'''<wfs:boundedBy>
                    <gml:Box srsName="EPSG:{srid}">
                        <gml:coordinates>{",".join([str(val) for val in bbox])}</gml:coordinates>
                    </gml:Box>
                </wfs:boundedBy>
                '''

            if layer_name == 'image-outlines':
                query_str = sql.SQL('''
                    SELECT img.id AS img_id, img.filename AS file_name,
                        img.x AS img_x, img.y AS img_y,
                        img.width AS img_width, img.height AS img_height,
                        img.affine_transform, img.extent
                    FROM {id_img} AS img
                    {bbox_sql}
                    {count_str};
                ''').format(
                    id_img=sql.Identifier(project, 'image'),
                    bbox_sql=sql.SQL(bbox_sql),
                    count_str=sql.SQL(count_str)
                )
            else:
                username_sql = ''
                if layer_name == 'annotation' and len(usernames) > 0:
                    # subset visible features to those made by specified users
                    if len(bbox_sql) > 0:
                        username_sql = 'AND meta.username IN ({})'
                    else:
                        username_sql = 'WHERE meta.username IN ({})'
                    username_sql = username_sql.format(
                        ','.join('%s' for _ in range(len(usernames)))
                    )
                    query_args.extend(usernames)
                query_str = sql.SQL('''
                    SELECT img.id AS img_id, img.filename AS file_name,
                            img.width AS img_width, img.height AS img_height,
                            img.affine_transform, img.extent,
                            meta.*, lc.name AS labelclass
                    FROM {id_img} AS img
                    JOIN {id_meta} AS meta
                    ON img.id = meta.image
                    JOIN {id_labelclass} AS lc
                    ON meta.label = lc.id
                    {bbox_sql}
                    {username_sql}
                    {count_str};
                ''').format(
                    id_img=sql.Identifier(project, 'image'),
                    id_meta=sql.Identifier(project, layer_name),
                    id_labelclass=sql.Identifier(project, 'labelclass'),
                    bbox_sql=sql.SQL(bbox_sql),
                    username_sql=sql.SQL(username_sql),
                    count_str=sql.SQL(count_str)
                )
            if count > 0:
                query_args.append(count)
            features = self.db_connector.execute(query_str,query_args, 'all')
            if features is not None:
                for feature in features:
                    # pre-populate custom metadata fields
                    if layer_name == 'image-outlines':
                        feature['id'] = str(feature['img_id'])
                        feature['link'] = f'{base_url}/{project}/interface?imgs={feature["img_id"]}'
                        filename = f'{base_url}/{project}/files/{feature["file_name"]}'
                        if all(feature[key] is not None for key in ('img_x', 'img_y')):
                            filename += '?window=' + \
                                ','.join(f'{feature[key]}' for key in \
                                    ('img_x', 'img_y', 'img_width', 'img_height'))
                        feature['file_link'] = filename

                    gml_features += self._encode_features(feature,
                                                            layer_name,
                                                            type_name,
                                                            version,
                                                            srid,
                                                            flip_coordinates)

        format_args = {
            'gml_bbox': bbox_gml,       #TODO: make layer-specific
            'features': gml_features
        }
        return self.render_service_template(version, 'get_feature', format_args, True), \
                self.DEFAULT_RESPONSE_HEADERS


    def _load_service_requests(self):
        self._register_service_request('GetCapabilities', self.get_capabilities)
        self._register_service_request('DescribeFeatureType', self.describe_feature_type)
        self._register_service_request('GetFeature', self.get_feature)
