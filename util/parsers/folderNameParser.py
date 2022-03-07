'''
    Label folder name parser. Extracts the topmost or one of the topmost folder
    names of a list of image files and uses them for label classes. The precise
    hierarchy level used depends on a combination of image file names as
    registered in the database, correlation of retrieved folder names with
    existing label classes in the project, as well as the number of unique
    folder names retrieved (should be minimal).

    2022 Benjamin Kellenberger
'''

import os
from psycopg2 import sql
from util.parsers.abstractParser import AbstractAnnotationParser


class FolderNameParser(AbstractAnnotationParser):

    ANNOTATION_TYPES = ('labels')

    def _determine_folder_level(self, fileList):
        '''
            Attempts to retrieve a common level of folder hierarchy based on a
            given list of file names and returns the level (root = index zero)
            that is the best match. Matches are determined based on two factors:
            1. number of matches of folder names against existing label classes
               in project (as high as possible);
            2. number of unique folder names at given level (as low as possible,
               but never just 1).

            Returns None if no common and suitable level could be found (e.g.,
            in the case of image names only).
            Also returns a list of folders. This is rather ugly from a coding
            perspective, but it reduces redundant computations during import
            later on.
        '''
        # split file parent directories into folder tokens
        folders = []
        for f in fileList:
            parent, _ = os.path.split(f)
            parent = parent.strip(os.sep)
            if not len(parent):
                folders.append([])
            else:
                folders.append(parent.split(os.sep))

        # determine most appropriate level in folder hierarchy
        bestLevel = -1
        bestNumMatches = 0      # number of label class name matches with existing for best level (as large as possible)
        bestNumDistinct = 1e9   # number of distinct folder names for best level (as small as possible, but more than one)

        currentLevel = 0
        while True:
            keys = set([f[currentLevel] for f in folders if currentLevel < len(f)])
            if len(keys) == 0:
                break
            elif len(keys) > 1:
                if len(self.labelClasses):
                    # test against existing label classes
                    numMatches = len(keys.intersection(set(self.labelClasses.keys())))
                    if numMatches > bestNumMatches:
                        bestLevel = currentLevel
                        bestNumMatches = numMatches
                        bestNumDistinct = len(keys)
                        break
                    elif len(keys) < bestNumDistinct and currentLevel == 0:
                        # no intersection with existing label classes; check number of distinct keys
                        # we only do it if we haven't already found a sensible folder (priority for high levels)
                        bestLevel = currentLevel
                        bestNumDistinct = len(keys)
                else:
                    # no existing label classes; check number of distinct keys
                    if len(keys) < bestNumDistinct:
                        bestLevel = currentLevel
                        bestNumDistinct = len(keys)
                        break       # no label classes present: we take the topmost sensible level
            currentLevel += 1

        if bestLevel < 0:
            return None, None
        else:
            # extract class names
            classNames = []
            for f in fileList:
                tokens = f.strip(os.sep).split(os.sep)
                if bestLevel < len(tokens):
                    classNames.append(tokens[bestLevel])
                else:
                    classNames.append(None)
            return bestLevel, classNames


    def is_parseable(self, fileList):
        '''
            File list is parseable if a clear folder structure with sensible
            label class names can be found.
        '''
        return self._determine_folder_level(fileList)[0] is not None
    

    def import_annotations(self, fileList, targetAccount, skipUnknownClasses, markAsGoldenQuestions, **kwargs):
        warnings = []

        # verify files for validity
        fileList_filtered = []
        for f in fileList:
            if os.path.isdir(f):
                warnings.append(f'"{f}" is a directory.')
            elif not os.path.isfile(f) and not os.path.islink(f):
                warnings.append(f'"{f}" could not be found on disk.')
            else:
                fileList_filtered.append(f)
        fileList = fileList_filtered

        # get current label class names, optimal folder hierarchy level, and folder tokens
        bestLevel, classNames = self._determine_folder_level(fileList)

        if bestLevel < 0 or classNames is None:
            # no common folder level found; abort
            return {
                'ids': [],
                'warnings': warnings,
                'errors': ['No common folder hierarchy level found for valid class names.']
            }
        
        # find matching images for file names
        imgs_match = self.match_filenames(fileList)

        # filter valid files: found in database and with valid label
        imgs_valid = []
        lc_new = set()          # new label classes to register in project
        for idx, img in enumerate(imgs_match):
            if img is None:
                warnings.append(
                    f'"{fileList[idx]}" could not be found in project.'
                )
                continue

            # extract label class name based on level
            className = classNames[idx]
            if className is None:
                warnings.append(f'"{fileList[idx]}": no label class name found.')
                continue
            elif className not in self.labelClasses:
                if skipUnknownClasses:
                    warnings.append(f'"{fileList[idx]}": class "{className}" not present in project.')
                    continue
                else:
                    lc_new.add(className)

            # image exists and class name is valid
            imgs_valid.append([img[0], className])      # image ID, class name
        
        # add new label classes
        if not skipUnknownClasses and len(lc_new):
            self.dbConnector.insert(sql.SQL('''
                INSERT INTO {} (name)
                VALUES %s;
            ''').format(sql.Identifier(self.project, 'labelclass')),
            [(l,) for l in lc_new])

            # update LUT
            self._init_labelclasses()

        if len(imgs_valid):
            # replace label class names with IDs
            for l in range(len(imgs_valid)):
                className = imgs_valid[l][1]
                imgs_valid[l][1] = self.labelClasses[className]

            # finally, add annotations to database
            result = self.dbConnector.insert(sql.SQL('''
                INSERT INTO {} (image, label, username)
                VALUES %s
                RETURNING id, image;
            ''').format(sql.Identifier(self.project, 'annotation')),
            [(i[0], i[1], targetAccount) for i in imgs_valid], 'all')
            ids_inserted = [r[0] for r in result]

            # mark as golden questions if needed
            if markAsGoldenQuestions:
                ids_image = set([r[1] for r in result])
                self.dbConnector.execute(sql.SQL('''
                    UPDATE {}
                    SET isGoldenQuestion = TRUE
                    WHERE id IN %s;
                ''').format(sql.Identifier(self.project, 'image')),
                (tuple(ids_image),), None)

        return {
            'ids': ids_inserted,
            'warnings': warnings,
            'errors': []
        }


#TODO
if __name__ == '__main__':
    from util.configDef import Config
    from modules.Database.app import Database
    cfg = Config()
    dbConn = Database(cfg)
    parser = FolderNameParser(cfg, dbConn, 'labels')

    import glob
    fileList = glob.glob('/data/aide/projects/labels/**/*', recursive=True)

    parser.import_annotations(fileList, 'bkellenb', False, True)