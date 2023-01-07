'''
    2023 Benjamin Kellenberger
'''

ID_DELIM = '_'

METADATA_SPEC = {
    'annotation': {
        'id': 'string',
        'username': 'string',
        'labelclass': 'string',
        'link': 'string',
        'meta': 'string',
        'autoconverted': 'boolean',
        'timecreated': 'dateTime',
        'timerequired': 'float',
        'unsure': 'boolean'
    },
    'prediction': {
        'id': 'string',
        'labelclass': 'string',
        'link': 'string',
        'cnnstate': 'string',
        'confidence': 'float',
        'priority': 'float',
        'meta': 'string',
        'autoconverted': 'boolean',
        'timecreated': 'dateTime',
        'timerequired': 'float',
        'unsure': 'boolean'
    },
    'image-outlines': {
        'id': 'string',
        'link': 'string',
        'file_name': 'string',
        'file_link': 'string'
    }
}
