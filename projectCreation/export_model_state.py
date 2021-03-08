'''
    Retrieves the latest model state and saves it into a .pth file
    (essentially a pickled state dict) that can be loaded with the
    respective model function.

    2019 Benjamin Kellenberger
'''

import os
import io
import argparse
import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retrieve latest CNN model state and save to disk.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    parser.add_argument('--target_file', type=str, default='model_state.pth', const=1, nargs='?',
                    help='Target filename for the model.')
    args = parser.parse_args()


    # setup
    print('Setup...')
    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)

    from util.configDef import Config
    from modules import Database

    config = Config()


    # setup DB connection
    dbConn = Database(config)
    if not dbConn.canConnect():
        raise Exception('Error connecting to database.')
    dbSchema = config.getProperty('Database', 'schema')


    # get state dict
    print('Retrieving model state...')
    stateDict_raw = dbConn.execute('SELECT statedict FROM {schema}.cnnstate WHERE partial IS FALSE ORDER BY timecreated DESC LIMIT 1;'.format(schema=config.getProperty('Database', 'schema')), None, 1)

    
    # convert from bytes and save to disk
    print('Saving model state...')
    stateDict_parsed = io.BytesIO(stateDict_raw[0]['statedict'])
    stateDict_parsed = torch.load(stateDict_parsed, map_location=lambda storage, loc: storage)
    torch.save(stateDict_parsed, open(args.target_file, 'wb'))