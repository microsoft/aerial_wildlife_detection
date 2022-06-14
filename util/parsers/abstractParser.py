'''
    Abstract base class for annotation parsers.

    2022 Benjamin Kellenberger
'''

import os
import tempfile
from psycopg2 import sql



class AbstractAnnotationParser:

    NAME = '(abstract parser)'  # format name
    INFO = ''                   # HTML-formatted info text about format
    ANNOTATION_TYPES = ()       # annotation types supported by parser

    def __init__(self, config, dbConnector, project, user, annotationType):
        self.config = config
        self.dbConnector = dbConnector
        self.project = project
        self.user = user
        self.annotationType = annotationType
        assert annotationType in self.ANNOTATION_TYPES, f'unsupported annotation type "{annotationType}"'

        self.projectRoot = os.path.join(self.config.getProperty('FileServer', 'staticfiles_dir'), self.project)
        self.tempDir = os.path.join(
            self.config.getProperty('FileServer', 'tempfiles_dir', type=str, fallback=tempfile.gettempdir()),
            self.project
        )

        self._init_labelclasses()


    def _init_labelclasses(self):
        '''
            Creates a dict of name : id of all the label classes present in
            the project.
        '''
        # create project label class LUT
        lcIDs = self.dbConnector.execute(sql.SQL('''
            SELECT id, name
            FROM {};
        ''').format(sql.Identifier(self.project, 'labelclass')),
        None, 'all')
        self.labelClasses = dict([[l['name'], l['id']] for l in lcIDs])
    

    @classmethod
    def get_html_options(cls, method):
        '''
            Returns a string for a HTML block containing elements for
            parser-specific properties (if available, else just returns an empty
            string).
            Parameter "method" can be one of {'import', 'export'} and determines
            the mode (label import or export) for which the HTML options string
            should be prepared.
            During parsing, only "input" HTML elements (all types) will be checked
            within the block returned here and a JSON object with "input" elements'
            id as key and value will be provided to the parser.
        '''
        return ''


    @classmethod
    def is_parseable(cls, fileList):
        '''
            Receives an Iterable of strings of files and determines whether the
            provided files correspond to the format expected by the parser.
            Returns True or False, respectively.
        '''
        raise NotImplementedError('Not implemented for abstract base class.')


    def import_annotations(self, fileList, targetAccount, skipUnknownClasses, markAsGoldenQuestions, **kwargs):
        '''
            Inputs:
            - "fileList"        Iterable of strings denoting the files that are
                                to be parsed. Any annotation parser must be able
                                to cope with files that are not necessary for its
                                job.
            - "targetAccount"   Account shortname under which successfully parsed
                                annotations are to be inserted into the database.
                                Throws an error if account cannot be found or is
                                unauthorized for the current project.
            - "skipUnknownClasses"  If True, classes that cannot be found in tar-
                                    get project will be ignored. The precise i-
                                    dentification of unknown classes depends on
                                    the parser and format of the annotations.
            - "markAsGoldenQuestions"   If True, imported annotations will be marked
                                        as golden questions.
            Parsers may accept any custom keyword arguments for options.

            Returns a dict with entries as follows:
                {
                    'result': Dict of <image UUID> : Iterable of successfully imported annotation UUIDs
                    'warnings': Iterable of strings denoting warnings that occurred
                                during import,
                    'errors': Iterable of strings denoting errors/unparsable
                              annotations *
                }
            
            * note that this will only include annotations that are, in principle,
              parseable, but could not be imported due to unforeseen reasons. It
              does NOT include data (files, etc.) that are unreadable by the cur-
              rent parser.
        '''
        raise NotImplementedError('Not implemented for abstract base class.')

    
    def export_annotations(self, annotations, destination, **kwargs):
        '''
            Inputs:
            - "annotations":    Dict as follows:
                                {
                                    "labelclasses": [],
                                    "images": [],
                                    "annotations": [
                                        {
                                            "id": <UUID>,
                                            "labelclass": <UUID>,
                                            # any other, annotation type-specific fields
                                        }
                                    ]
                                }
            - "destination":    Zipfile handle to write files in. The exact file(s)
                                created therein is up to the parser subclass.
            Parsers may accept any custom keyword arguments for options.

            Returns a dict with entries as follows:
                {
                    'files':    Iterable of files created in the "destination"
                }
        '''
        raise NotImplementedError('Not implemented for abstract base class.')
    
    
    def match_filenames(self, fileNames, bidirectional=False):
        '''
            Receives an Iterable of file names (e.g., 'folder/file.jpg') and
            returns a list of equivalent image IDs and full file names as found
            in the database.
            "bidirectional" determines how the matching is performed:
            - if False, file names must match exactly
            - if True, matching is performed with bidirectional wildcard
                prefix search - for example, the following file names would match:

                    query: 'train/folder/file.jpg'
                    database: 'folder/file.jpg'
                
                as would the inverse case:

                    query: 'folder/file.jpg'
                    database: 'train/folder/file.jpg'
                Note that this is extremely slow.
            
            Any unidentifiable pattern will result in a value of None.
        '''
        if bidirectional:
            result = self.dbConnector.insert(sql.SQL('''
                SELECT img.id AS id, img.filename AS filename, m.pat AS pat
                FROM {} AS img
                JOIN (VALUES %s) AS m(pat)
                ON m.pat LIKE CONCAT('%%', img.filename) OR img.filename LIKE CONCAT('%%', m.pat);
            ''').format(sql.Identifier(self.project, 'image')),
            [(f,) for f in fileNames], 'all')
            result = dict(zip([r[2] for r in result], [(r[0], r[1]) for r in result]))
        
        else:
            result = self.dbConnector.insert(sql.SQL('''
                SELECT img.id AS id, img.filename AS filename
                FROM {} AS img
                WHERE filename IN %s;
            ''').format(sql.Identifier(self.project, 'image')),
            (fileNames,), 'all')
            result = dict(zip([r[1] for r in result], [(r[0], r[1]) for r in result]))

        response = []
        for f in fileNames:
            response.append(result.get(f, None))
        return response