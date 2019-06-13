'''
    Wrappers for different types of login errors.

    2019 Benjamin Kellenberger
'''
class InvalidRequestException(Exception):
    def __init__(self):
        super(InvalidRequestException, self).__init__('invalid request')

class ValueMissingException(Exception):
    def __init__(self, valueName):
        super(ValueMissingException, self).__init__('{} required but not provided'.format(valueName))

class InvalidPasswordException(Exception):
    def __init__(self):
        super(InvalidPasswordException, self).__init__('invalid password provided')

class SessionTimeoutException(Exception):
    def __init__(self):
        super(SessionTimeoutException, self).__init__('session time-out')

class AccountExistsException(Exception):
    def __init__(self, username):
        super(AccountExistsException, self).__init__('{} already exists'.format(username))