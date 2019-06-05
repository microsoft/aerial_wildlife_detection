'''
    Runs one of the two tiers (LabelingUI or AItrainer), based on the arguments passed.

    2019 Benjamin Kellenberger
'''

import argparse
from configparser import ConfigParser
from modules.LabelUI.app import LabelUI





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run CV4Wildlife AL Service.')
    parser.add_argument('--instance', type=str, default='frontend', const=1, nargs='?',
                    help='Which instance type to run on this host. One of {"frontend", "aitrainer"} (default: "frontend").')
    args = parser.parse_args()


    # read configuration
    config = ConfigParser()
    config.read('config/settings.ini')

    if args.instance == 'frontend':
        LabelUI(config)
    
    else:
        raise ValueError('{} is not a recognized type of instance.'.format(args.instance))