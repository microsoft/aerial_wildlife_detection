'''
    2023 Benjamin Kellenberger
'''

from typing import Iterable

from util.helpers import DEFAULT_RENDER_CONFIG

SERVICE_VERSIONS = {
        'wms': [
            '1.3.0'
        ],
        'wcs': [
            '1.1.0'
        ],
        'wfs': [
            '2.0.0'
        ]
    }

DEFAULT_SERVICE_VERSIONS = {
    'wms': '1.3.0',
    'wcs': '1.1.0',
    'wfs': '2.0.0'
}

DEFAULT_MAPSERVER_SETTINGS = {
    'enabled': False,
    'layers': {
        'image-outlines': {
            'name': 'Image outlines',
            'services': {
                'wfs': {
                    'enabled': False,
                    'acl': {
                        'non_admin': False
                    }
                }
            }
        },
        'images': {
            'name': 'Images',
            'services': {
                'wms': {
                    'enabled': False,
                    'acl': {
                        'non_admin': False
                    },
                    'options': {
                        'render_config': DEFAULT_RENDER_CONFIG
                    }
                },
                'wcs': {
                    'enabled': False,
                    'acl': {
                        'non_admin': False
                    }
                }
            }
        },
        'annotation': {
            'name': 'User annotations',
            'services': {
                'wms': {
                    'enabled': False,
                    'acl': {
                        'non_admin': False
                    }
                },
                'wcs': {
                    'enabled': False,
                    'acl': {
                        'non_admin': False
                    }
                },
                'wfs': {
                    'enabled': False,
                    'acl': {
                        'non_admin': False
                    }
                }
            }
        },
        'prediction': {
            'name': 'Model predictions',
            'services': {
                'wms': {
                    'enabled': False,
                    'acl': {
                        'non_admin': False
                    }
                },
                'wcs': {
                    'enabled': False,
                    'acl': {
                        'non_admin': False
                    }
                },
                'wfs': {
                    'enabled': False,
                    'acl': {
                        'non_admin': False
                    }
                }
            }
        }
    }
}

def find_version(service_type: str, versions: Iterable) -> str:
    '''
        Receives a service type and template versions and returns the most fitting one.
        Inputs:
        - "service_type": str, type of service (e.g., "WMS", "WCS")
        - "versions": Iterable, suggested version (str) or multiple suggestions

        Returns:
        str, optimal version based on arguments or None if impossible to resolve.
    '''
    service_type = service_type.lower()
    if service_type not in SERVICE_VERSIONS:
        return None
    service_group = SERVICE_VERSIONS[service_type]
    if versions is None:
        return DEFAULT_SERVICE_VERSIONS.get(service_type, None)
    if isinstance(versions, str):
        versions = (versions,)
    for version_test in versions:
        if version_test in service_group:
            return version_test
    return DEFAULT_SERVICE_VERSIONS.get(service_type, None)
