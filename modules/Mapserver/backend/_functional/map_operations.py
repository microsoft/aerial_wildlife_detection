'''
    Raster (image, segmentation mask) retrieval, stitching, and rendering functionalities.

    2023 Benjamin Kellenberger
'''

import os
import numpy as np
from psycopg2 import sql
import rasterio
import rasterio.merge

from modules.Database.app import Database
from util import helpers


def render_array(array: np.array,
                render_config: dict,
                to_uint8: bool=False) -> np.array:
    '''
        Applies the properties specified in "render_config" to a given NumPy ndarray, including:
        - grayscale conversion (if set)
        - white on black (if set)
        - contrast enhancement (if set)
        - brightness adjustment (if set)
        Returns the rendered array accordingly.
    '''
    # grayscale
    if render_config.get('grayscale', False):
        array = np.mean(array, 0, keepdims=True)
    # band-specific
    multiplier = 255 if to_uint8 else 1
    for band in range(len(array)):
        if render_config.get('white_on_black', False):
            array[band,...] = np.max(array[band,...]) - array[band,...]
        if 'percentile' in render_config.get('contrast', {}):
            percentiles = render_config['contrast']['percentile']
            band_min = np.percentile(array[band,...], percentiles['min'])
            band_max = np.percentile(array[band,...], percentiles['max'])
            array[band,...] = multiplier * (array[band,...] - band_min) / (band_max - band_min)
        elif to_uint8:
            band_min, band_max = np.min(array[band,...]), np.max(array[band,...])
            array[band,...] = multiplier * (array[band,...] - band_min) / (band_max - band_min)
    array += render_config.get('brightness', 0)
    if to_uint8:
        array = array.astype(np.uint8)
    return array


def get_map_images(db_connector: Database,
                    images_dir: str,
                    project: str,
                    project_meta: dict,
                    bbox: tuple,
                    srid: int,
                    resolution: tuple,
                    image_ext: str,
                    raw: bool=False) -> bytes:
    '''
        TODO
    '''
    # determine bands to extract (in case of multi-band images)
    bands = None
    if not raw:
        bands = [1, 2, 3]
        render_config = project_meta['render_config']
        if 'indices' in render_config.get('bands', {}):
            indices = render_config['bands']['indices']
            if 'grayscale' in indices:
                bands = [indices['grayscale']+1]
            else:
                bands = [
                    indices['red']+1,
                    indices['green']+1,
                    indices['blue']+1
                ]

    # find all images that intersect
    query_str = sql.SQL('''
        SELECT DISTINCT filename
        FROM {id_img} AS img
        WHERE ST_Intersects(
            img.extent,
            ST_MakeEnvelope(
                    %s, %s, %s, %s,
                    %s
            )
        );
    ''').format(
        id_img=sql.Identifier(project, 'image')
    )
    imgs = db_connector.execute(query_str,
        (*bbox, srid), 'all')
    rio_datasets = []
    if len(imgs) > 0:
        try:
            for img in imgs:
                img_path = os.path.join(images_dir, project, img['filename'])
                rio_datasets.append(rasterio.open(img_path, 'r'))

            # merge
            arr, transform = rasterio.merge.merge(datasets=rio_datasets,
                                                    bounds=bbox,
                                                    res=resolution,
                                                    indexes=bands)

            # rescale and convert if necessary
            if not raw:
                arr = render_array(arr, render_config, to_uint8=True)

            # save to memfile and return
            meta = {
                'driver': rasterio.driver_from_extension(image_ext),
                'count': arr.shape[0],
                'width': arr.shape[2],
                'height': arr.shape[1],
                'dtype': str(arr.dtype),
                'transform': transform,
                'crs': rasterio.crs.CRS.from_epsg(srid)
            }
            with rasterio.MemoryFile() as memfile:
                with memfile.open(**meta) as f_raster:
                    f_raster.write(arr)
                memfile.seek(0)
                return bytes(memfile.getbuffer())
        finally:
            for dataset in rio_datasets:
                dataset.close()
    else:
        # no image intersects with queried extent
        return None     #TODO


def get_map_segmentation(db_connector: Database,
                            images_dir: str,
                            project: str,
                            project_meta: dict,
                            relation_name: str,
                            username: str,
                            bbox: tuple,
                            srid: int,
                            resolution: tuple,
                            image_ext: str,
                            raw: bool=False) -> bytes:
    '''
        TODO
    '''
    assert relation_name in ('annotation', 'prediction')

    # find all images and segmentation masks that intersect
    query_str = sql.SQL('''
        SELECT meta.segmentationmask, img.filename
        FROM {id_img} AS img
        JOIN {id_meta} AS meta
        ON img.id = meta.image
        WHERE meta.username = %s AND ST_Intersects(
            img.extent,
            ST_MakeEnvelope(
                    %s, %s, %s, %s,
                    %s
            )
        );
    ''').format(
        id_img=sql.Identifier(project, 'image'),
        id_meta=sql.Identifier(project, relation_name)
    )
    meta = db_connector.execute(query_str,
        (username, *bbox, srid), 'all')
    if len(meta) == 0:
        return None     #TODO

    driver = rasterio.driver_from_extension(image_ext)

    label_classes = project_meta.get('label_classes', [])

    # decode segmentation masks and wrap in rasterio DatasetReader instances
    rio_datasets = []
    try:
        for item in meta:

            # convert base64 mask to array, then to rasterio Memfile
            img_path = os.path.join(
                images_dir, project, item['filename']
            )
            if not os.path.exists(img_path):
                continue
            with rasterio.open(img_path, 'r') as f_img:
                profile = f_img.profile
            raster = helpers.base64ToImage(item['segmentationmask'],
                                            profile['width'], profile['height'], False)
            profile.update({
                'driver': 'GTiff',
                'dtype': 'uint8',
                'nodata': 0,
                'count': 1
            })
            with rasterio.MemoryFile() as memfile:
                with rasterio.open(memfile, 'w', **profile) as f_mem:
                    f_mem.write(raster, 1)
                rio_datasets.append(memfile.open())

        if len(rio_datasets) == 0:
            return None     #TODO

        # merge
        arr, transform = rasterio.merge.merge(datasets=rio_datasets,
                                                bounds=bbox,
                                                res=resolution)

        if not raw:
            # WMS: render as RGB
            if len(label_classes) > 0:
                arr_rgb = np.zeros((3, *arr.shape[1:]), dtype=rasterio.ubyte)
                for label_class in label_classes:
                    valid = (arr == label_class['idx']).squeeze()
                    for band in range(3):
                        arr_rgb[band,valid] = label_class['color'][band]
                arr = arr_rgb

        # save to new memfile and return
        meta = {
            'driver': driver,
            'count': arr.shape[0],
            'width': arr.shape[2],
            'height': arr.shape[1],
            'dtype': 'uint8',
            'transform': transform,
            'crs': rasterio.crs.CRS.from_epsg(srid),
            'nodata': 0
        }

        with rasterio.MemoryFile() as memfile:
            with memfile.open(**meta) as f_raster:
                f_raster.write(arr)
                if raw:
                    # WCS: write colormap
                    color_map = dict([item['idx'], item['color']] for item in label_classes)
                    if len(color_map) > 0:
                        f_raster.write_colormap(1, color_map)

            memfile.seek(0)
            return bytes(memfile.getbuffer())
    finally:
        for dataset in rio_datasets:
            dataset.close()
