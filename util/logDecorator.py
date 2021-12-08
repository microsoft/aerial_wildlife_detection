'''
    Class for pretty-printing startup messages and statuses to the command line.

    2021 Benjamin Kellenberger
'''

import os


class LogDecorator:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def get_ljust_offset():
        try:
            return os.get_terminal_size().columns - 6
        except:
            return 74

    @staticmethod
    def print_status(status, color=None):
        if status.lower() == 'ok':
            print(f'{LogDecorator.OKGREEN}[ OK ]{LogDecorator.ENDC}')
        elif status.lower() == 'warn':
            print(f'{LogDecorator.WARNING}[WARN]{LogDecorator.ENDC}')
        elif status.lower() == 'fail':
            print(f'{LogDecorator.FAIL}[FAIL]{LogDecorator.ENDC}')
        else:
            if color is not None:
                print(f'{getattr(LogDecorator, color)}[{status}]{LogDecorator.ENDC}')
            else:
                print(f'[{status}]')