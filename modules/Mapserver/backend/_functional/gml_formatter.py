'''
    Helper class to format geometries and metadata as GML strings for Web Mapserver services.

    2023 Benjamin Kellenberger
'''

import os
from typing import Iterable
from decimal import Decimal

from .. import METADATA_SPEC
from ... import SERVICE_VERSIONS, find_version


class GMLFormatter:

    def __init__(self):
        # load templates
        self.geometry_templates = {}
        for version in SERVICE_VERSIONS['wfs']:
            self.geometry_templates[version] = {}
            for geom_type in ('point', 'polygon'):
                template_path = os.path.abspath(
                    os.path.join(
                        'modules/Mapserver/static/xml',
                        'wfs', version, f'gml_{geom_type}.xml'
                    )
                )
                with open(template_path, 'r', encoding='utf-8') as f_template:
                    self.geometry_templates[version][geom_type] = f_template.read()


    def _encode_metadata(self, meta: dict, annotation_type: str) -> str:
        meta_str = ''
        for key in METADATA_SPEC[annotation_type].keys():
            val = meta.get(key, None)
            if val is None:
                val = ''
            elif isinstance(val, Decimal):
                val = float(val)
            elif isinstance(val, (bool)):
                pass
            else:
                val = str(val)
            meta_str += f'<{key}>{val}</{key}>'
        return meta_str


    def _encode_geometry(self, geometry: dict, version: str) -> str:
        if all(key in geometry for key in ('x', 'y')) and \
            not any(key in geometry for key in ('width', 'height')):
            # point
            return self.geometry_templates[version]['point'].format_map(geometry)
        # polygon
        return self.geometry_templates[version]['polygon'].format_map(geometry)


    def to_gml(self, meta: Iterable,
                    annotation_type: str,
                    type_name: str,
                    version: str=None) -> str:
        '''
            Receives metadata (dict or Iterable of dicts) and returns a GML-compliant XML string
            with their properties encoded.
        '''
        version_lookup = find_version('wfs', version)
        assert version_lookup in self.geometry_templates, \
            f'ERROR: unsupported service version "{version}"'

        # encode metadata
        if isinstance(meta, dict):
            meta = (meta,)
        result = ''
        for item in meta:
            # geometry
            gml_geom = self._encode_geometry(item, version_lookup)

            # metadata fields
            gml_metadata = self._encode_metadata(item, annotation_type)

            gml_item = '''
                <wfs:member xmlns:wfs="http://www.opengis.net/wfs/2.0">
                    <{type_name} xmlns:gml="http://www.opengis.net/gml" gml:id="{item_id}">
                        {gml_geom}
                        {gml_metadata}
                    </{type_name}>
                 </wfs:member>
            '''.format(
                type_name=type_name,
                item_id=str(item['id']),
                gml_geom=gml_geom,
                gml_metadata=gml_metadata
            )
            result += gml_item
        return result
