'''
    Little helper that commits a PyTorch model state from a local path to the DB.

    2019 Benjamin Kellenberger
'''

import os
import io
import argparse
import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load model state from disk and commit to DB.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    parser.add_argument('--modelPath', type=str,
                    help='Directory (absolute path) on this machine of the model state file to be considered.')
    args = parser.parse_args()
    

    # setup
    print('Setup...')
    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)

    from util.configDef import Config
    from modules.Database.app import Database
    config = Config()
    dbConn = Database(config)


    # load model class function
    print('Load and verify state dict...')
    from util.helpers import get_class_executable, current_time
    modelClass = getattr(get_class_executable(config.getProperty('AIController', 'model_lib_path')), 'model_class')

    # load state dict
    stateDict = torch.load(open(args.modelPath, 'rb'))

    # verify model state
    model = modelClass.loadFromStateDict(stateDict)

    # load class definitions from database
    classdef_db = {}
    labelClasses = dbConn.execute('SELECT * FROM {schema}.labelclass;'.format(schema=config.getProperty('Database', 'schema')), None, 'all')
    for lc in labelClasses:
        classdef_db[lc['id']] = lc
    classIDs_db = set(classdef_db.keys())


    # set class definition
    confirmation = 'n'
    forceManualAssignment = False   # if True (e.g. after confirmation), all classes need to be manually assigned, even if a match exists

    while confirmation == 'n':

        # target classdef
        classdef_final = {}

        # compare with current state
        if 'labelclassMap' in stateDict:
            print('Found label class definition (name to index map)...')
            classdef_state = stateDict['labelclassMap']
            classIDs_state = set(classdef_state.keys())

            intersect = list(classIDs_db.intersection(classIDs_state))
            if not forceManualAssignment:
                for ii in intersect:
                    print('Found common labelclass {}: {}'.format(classdef_state[ii], ii))
                    classdef_final[ii] = classdef_state[ii]

            setdiff_left = list(classIDs_db.difference(classIDs_state))
            setdiff_right = list(classIDs_state.difference(classIDs_db))

            if len(setdiff_right):
                print('Found model prediction branch(es) that cannot automatically be assigned to a label class.')
                print('You will have to manually provide a map between model prediction indices and label class names.')

                # put them in index order
                order = [classdef_state[x] for x in setdiff_right if x not in intersect]
                order.sort()
                classdef_state_inv = {}
                for key in classdef_state.keys():
                    classdef_state_inv[classdef_state[key]] = key

                for o in order:
                    if not len(setdiff_left):
                        print('No more classes left in database; skipping remaining prediction classes...')
                        break

                    print('\nNext class in model state to be imported:\n')
                    print('INDEX\tNAME\n-----------------------')
                    print('{}\t{}'.format(o, classdef_state_inv[o]))
                    print('\n\nEnter the number of the target class this belongs to:\n')
                    for tt in range(len(setdiff_left)):
                        print('{}\t{}\n'.format(tt+1, classdef_db[setdiff_left[tt]]['name']))
                    
                    print('\n')
                    selection = None
                    while selection is None:
                        try:
                            selection = int(input('Enter Number: '))
                            if selection <= 0 or selection > len(setdiff_left):
                                raise Exception('out of range')
                        except:
                            selection = None
                    
                    # assign class
                    sel_UUID = setdiff_left[selection-1]
                    classdef_final[sel_UUID] = o
                    del setdiff_left[selection-1]

        else:
            # no class definition found; fallback to number of classes
            print('No label class definition (name to index map) found; you will have to manually assign classes.')

            if 'num_classes' in stateDict:
                numClasses = stateDict['num_classes']
            elif 'numClasses' in stateDict:
                numClasses = stateDict['numClasses']
            else:
                raise Exception('Failed to get number of classes from model state.')

            for nn in range(numClasses):
                if not len(setdiff_left):
                    print('No more classes left in database; skipping remaining prediction classes...')
                    break
                print('\nNext predicted index: {} (out of {})\n'.format(nn, numClasses))
                print('Enter the destination class index from the following selection:\n')
                print('INDEX\tNAME\n-----------------------')
                for tt in range(len(setdiff_left)):
                    print('{}\t{}\n'.format(tt+1, classdef_db[setdiff_left[tt]]['name']))

                print('\n')
                selection = None
                while selection is None:
                    try:
                        selection = int(input('Enter Number: '))
                        if selection <= 0 or selection > len(setdiff_left):
                            raise Exception('out of range')
                    except:
                        selection = None
                
                # assign class
                sel_UUID = setdiff_left[selection-1]
                classdef_final[sel_UUID] = nn
                del setdiff_left[selection-1]


        # review the final selection made
        print('The following model prediction index to label class assignment will be saved:\n')
        classdef_final_inv = {}
        for key in classdef_final.keys():
            classdef_final_inv[classdef_final[key]] = key
        
        for ii in range(len(classdef_final)):
            key = classdef_final_inv[ii]
            print('{}\t-->\t{}'.format(ii, classdef_db[key]['name']))
        
        print('Is this correct?')
        confirmation = None
        while confirmation is None:
            try:
                confirmation = input('[Y/n]: ')
                if 'Y' in confirmation:
                    confirmation = 'Y'
                elif 'n' in confirmation.lower():
                    confirmation = 'n'
                    forceManualAssignment = True
                else: raise Exception('Invalid value')
            except:
                confirmation = None


    # export model state
    print('Exporting model state...')
    bio = io.BytesIO()
    stateDict = model.getStateDict()

    # append new label class map definition
    stateDict['labelclassMap'] = classdef_final

    torch.save(stateDict, bio)
    stateDict = bio.getvalue()

    # commit to DB
    print('Committing to DB...')
    sql = '''
        INSERT INTO {schema}.cnnstate (timeCreated, stateDict, partial)
        VALUES ( %s, %s, FALSE);
    '''.format(schema=config.getProperty('Database', 'schema'))
    now = current_time()
    dbConn.execute(sql, (now, stateDict, ), None)