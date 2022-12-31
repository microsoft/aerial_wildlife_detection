'''
    Helpers for geospatial operations.

    2022 Benjamin Kellenberger
'''

from typing import Iterable
import pyproj
import rasterio
from modules.Database.app import Database
from util.drivers import GDALImageDriver


def get_srid(db_connector: Database,
            project: str) -> int:
    '''
        Inputs:
        - "db_connector": Database, instance
        - "project": str, project shortname

        Returns:
        - EPSG code for SRID or None if not defined (e.g., if PostGIS is not available)
    '''
    #TODO: check if column actually exists
    srid = db_connector.execute('''
        SELECT Find_SRID(%s, 'image', 'extent') AS srid;
    ''', (project,), 1)
    return srid[0]['srid'] if len(srid) > 0 else None


def to_crs(srid: object) -> pyproj.CRS:
    '''
        Receives an "srid", one of:
        - int: EPSG code
        - str: WKT definition of CRS
        - CRS: rasterio CRS object
        Returns a pyproj.CRS object
    '''
    if isinstance(srid, int):
        return pyproj.CRS.from_epsg(srid)
    if isinstance(srid, str):
        if srid.lower().startswith('+proj'):
            return pyproj.CRS.from_proj4(srid)
        return pyproj.CRS.from_string(srid)
    if isinstance(srid, rasterio.CRS):
        return pyproj.CRS.from_user_input(srid)
    return None


def calc_extent(file_name: str,
                srid: int=4326,
                window: tuple=None) -> tuple:
    '''
        Calculates the geospatial extent for an image (with optional window) and returns it as WKT
        string for insertion into PostGIS database.
        Inputs:
        - "file_name": str, path of image to calculate extent for
        - "srid": int, spatial reference id for output polygon
        - "window" tuple, optional window as (y, x, height, width)

        Returns:
        tuple, containing float values for (left, top, right, bottom) of extent
    '''
    try:
        meta = GDALImageDriver.metadata(file_name, window=window)
        crs_source = to_crs(meta.get('crs', None))
        if crs_source is None:
            return None
        bounds = meta.get('bounds', None)
        if bounds is None:
            return None
        crs_target = to_crs(srid)
        if not crs_source.equals(crs_target):
            # transform bounds
            transformer = pyproj.Transformer.from_crs(crs_source, crs_target,
                                                        always_xy=True)
            bounds = transformer.transform(*bounds)
        return bounds
    except Exception:
        return None
