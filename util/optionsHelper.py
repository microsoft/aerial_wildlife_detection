'''
    Various helper functions to deal with the GUI-enhanced,
    JSON-formatted AI and AL model options.
    Some of these functions are analogous to the "optionsEngine.js".

    2020 Benjamin Kellenberger
'''

import copy
from collections.abc import Iterable

from util import helpers


RESERVED_KEYWORDS = [
    'id', 'name', 'description', 'type', 'min', 'max', 'value', 'style', 'options'
]


def _flatten_globals(options, defs=None):
    '''
        Traverses a hierarchical dict of "options" and copies
        all sub-entries to the top level.
    '''
    if options is None:
        return
    if defs is None:
        defs = options.copy()
    if not isinstance(options, dict):
        return
    for key in options.keys():
        if key in RESERVED_KEYWORDS:
            continue
        elif isinstance(key, str):
            defs[key] = options[key]
            if isinstance(defs[key], dict) and not 'id' in defs[key]:   #not isinstance(options, Iterable):
                defs[key]['id'] = key
            _flatten_globals(options[key], defs)
    return defs



def _fill_globals(options, defs):
    '''
        Receives a dict of "options" that contains keywords
        under e.g. "value" keys, and searches the "defs"
        for substitutes.
        Replaces all entries in "options" in-place by corres-
        ponding entries in "defs", if present.
    '''
    if defs is None:
        return options
    if isinstance(options, str):
        if options in defs:
            return defs[options]
        else:
            # no match found
            return options

    elif isinstance(options, dict):
        # for lists and selects: add options to global definitions if not yet present
        if 'options' in options:
            if isinstance(options['options'], dict):
                for key in options['options'].keys():
                    if key not in defs:
                        defs[key] = options['options'][key]
                        if not 'id' in defs[key]:
                            defs[key]['id'] = key
            elif isinstance(options['options'], Iterable):
                for option in options['options']:
                    if isinstance(option, dict) and 'id' in option and not option['id'] in defs:
                        defs[option['id']] = option
                        if not 'id' in defs[option['id']]:
                            defs[defs[option['id']]]['id'] = option['id']

        keys = list(options.keys())
        for key in keys:
            if key in RESERVED_KEYWORDS and key != 'value':
                    continue
            elif isinstance(options[key], str):
                if options[key] in defs:
                    options[key] = defs[options[key]]
                elif 'options' in options and options[key] in options['options']:
                    options[key] = options['options'][key]
                if isinstance(options[key], dict) and not 'id' in options[key]:
                    options[key]['id'] = key
            elif isinstance(options[key], Iterable):
                options[key] = _fill_globals(options[key], defs)

    elif isinstance(options, Iterable):
        for idx, option in enumerate(options):
            if isinstance(option, str):
                if option in RESERVED_KEYWORDS and option != 'value':
                    continue
                elif option in defs:
                    options[idx] = defs[option]
                    if isinstance(options[idx], dict) and not 'id' in options[idx]:
                        options[idx]['id'] = option
            elif isinstance(option, dict):
                options[idx] = _fill_globals(options[idx], defs)
        
    return options



def substitute_definitions(options):
    '''
        Receives a dict of "options" with two main sub-dicts
        under keys "defs" (global definitions) and "options"
        (actual options for the model).
        First flattens the "defs" and substitutes values in
        the "defs" themselves, then substitutes values in
        "options" based on the "defs".
        Does everything on a new copy.
    '''
    if options is None:
        return None
    if not 'defs' in options or not 'options' in options:
        return options

    options_out = options.copy()
    defs = options_out['defs']
    defs = _flatten_globals(defs)
    defs = _fill_globals(defs, defs)
    options_out['options'] = _fill_globals(options_out['options'], defs)
    return options_out



def get_hierarchical_value(dictObject, keys, lookFor=('value', 'id'), fallback=None):
    '''
        Accepts a Python dict object and an iterable
        of str corresponding to hierarchically defined
        keys. Traverses the dict and returns the value
        provided under the key.

        "lookFor" can either be a str or an Iterable of
        str that are used for querying if the next key
        in line in "keys" cannot be found in the current
        (sub-) entry of "dictObject". For example, if
        "lookFor" is "('value', 'id')" and the current
        key in line is not present in the current
        "dictObject" sub-entry, that object will first be
        tried for 'value', then for 'id', and whichever
        comes first will be returned. If nothing could be
        found or in case of an error, the value specified
        under "fallback" is returned.

        Note also that this means that the ultimate key
        MUST be specified in the list of "keys", even if
        it is one of the reserved keywords (e.g. "id" or
        "value").
    '''
    try:
        #return reduce(dict.get, keys, dictObject)
        if not isinstance(dictObject, Iterable):
            return dictObject

        if not isinstance(keys, list):
            if not isinstance(keys, Iterable):
                keys = [keys]
            else:
                keys = list(keys)
        if not len(keys):
            return dictObject
            # if isinstance(lookFor, str) and lookFor in dictObject:
            #     return get_hierarchical_value(dictObject[lookFor], keys, lookFor)
            # elif isinstance(lookFor, Iterable):
            #     for keyword in lookFor:
            #         if keyword in dictObject:
            #             return get_hierarchical_value(dictObject[keyword], keys, lookFor)
            #     return dictObject
            # else:
            #     return dictObject
        if keys[0] in dictObject:
            key = keys.pop(0)
            return get_hierarchical_value(dictObject[key], keys, lookFor)
        elif isinstance(lookFor, str) and lookFor in dictObject:
            return get_hierarchical_value(dictObject[lookFor], keys, lookFor)
        elif isinstance(lookFor, Iterable):
            for keyword in lookFor:
                if keyword in dictObject:
                    return get_hierarchical_value(dictObject[keyword], keys, lookFor)
            return dictObject
        else:
            return fallback
    except:
        return fallback



def _append_current_hierarchy(dictObject, current, hierarchy=[]):
    # check current dictObject for children
    if isinstance(dictObject, dict):
        for key in dictObject.keys():
            if key in RESERVED_KEYWORDS:
                continue
            
    return hierarchy



def get_hierarchy(dictObject, substitute_globals=True):
    '''
        Receives a Python dict (options) object and returns
        a list of lists with IDs/names of each hierarchical
        level for all entries. Ignores entries under reser-
        ved keywords ('name', 'description', etc.).
        If "substitute_globals" is True, global definitions
        are parsed and added to the options prior to tra-
        versal.
    '''
    options = copy.deepcopy(dictObject)
    if substitute_globals:
        options = substitute_definitions(options)
    
    # iterate
    result = []
    #TODO


def set_hierarchical_value(dictObject, keys, value):
    '''
        Accepts a Python dict object and an iterable
        of str corresponding to hierarchically defined
        keys. Traverses the dict and updates the value
        at the final, hierarchical position provided
        under the keys with the new "value". Silently
        aborts if the keys could not be found.
        Modifications are done in-place, hence this
        method does not return anything.
    '''
    if not isinstance(keys, str) and len(keys) == 1:
        keys = keys[0]
    if isinstance(keys, str) and keys in dictObject:
        dictObject[keys] = value
    elif keys[0] in dictObject:
        set_hierarchical_value(dictObject[keys[0]], keys[1:], value)



def update_hierarchical_value(sourceOptions, targetOptions, sourceKeys, targetKeys):
    '''
        Retrieves a value from nested "sourceOptions" dict under the
        hierarchical list of "sourceKeys" and applies it under the
        hierarchical list of "targetKeys" in "targetOptions".
        Silently aborts if source and/or target keys/values are not
        existent.
    '''
    sourceVal = get_hierarchical_value(sourceOptions, sourceKeys)
    if sourceVal is None:
        return
    set_hierarchical_value(targetOptions, targetKeys, sourceVal)



def filter_reserved_children(options, recursive=False):
    '''
        Receives an "options" dict (might also be just a part)
        and returns a copy that only contains child entries whose
        ID is non-standard (i.e., not of one of the reserved
        keywords).
        If "recursive" is set to True, it also traverses through
        the tree and applies the same logic for all child elements,
        keeping only the "value" entry.
    '''
    if not isinstance(options, dict):
        return options
    
    response = {}
    for key in options.keys():
        if key == 'value' or key not in RESERVED_KEYWORDS:
            response[key] = filter_reserved_children(options[key], recursive)
        elif key in RESERVED_KEYWORDS:
            continue
        
    return response



def verify_options(options, autoCorrect=False):
    '''
        Receives an "options" dict (might also be just a part)
        and verifies whether the values are correct w.r.t. the
        indicated options type and value range. Skips verifica-
        tion of values whose validity cannot be determined.
        If "autoCorrect" is True, every value that can be cor-
        rected will be accordingly. For example, if an option
        is to be a numerical value and the value is beyond the
        min-max range, it will be clipped accordingly; bool va-
        lues will be parsed; etc.

        Returns:
            - "options": the input dict with values corrected
                         (if specified)
            - "warnings": a list of str describing any warnings
                          that were encountered during parsing
            - "errors": a list of str describing uncorrectable
                        errors that were encountered. If there
                        are any, the options can be considered
                        as invalid.
    '''
    warnings, errors = [], []

    if isinstance(options, dict):
        valueType = (options['type'] if 'type' in options else None)
        for key in options.keys():
            if key == 'value':
                if isinstance(options[key], dict):
                    # get available options
                    valid_IDs = set()
                    if 'options' in options:
                        for o in options['options']:
                            valid_IDs.add(get_hierarchical_value(o, ['id']))

                        thisID = get_hierarchical_value(options[key], ['id'])
                        if thisID not in valid_IDs:
                            errors.append(f'Selected option "{thisID}" for entry {key} could not be found in available options.')
                else:
                    value = helpers.toNumber(options[key])
                    if value is not None:
                        # numerical value
                        if valueType is not None and valueType != 'number':
                            errors.append(f'Expected {valueType} for entry {key}, got {type(options[key])}.') #TODO: key
                        else:
                            # check if there's a min-max range
                            if 'min' in options:
                                minVal = helpers.toNumber(options['min'])
                                if minVal is not None and value < minVal:
                                    warnings.append(f'Value "{value}" for entry {key} is smaller than specified minimum {minVal}.')
                                    if autoCorrect:
                                        value = minVal
                            if 'max' in options:
                                maxVal = helpers.toNumber(options['max'])
                                if maxVal is not None and value > maxVal:
                                    warnings.append(f'Value "{value}" for entry {key} is larger than specified maximum {maxVal}.')
                                    if autoCorrect:
                                        value = maxVal
                        options[key] = value

            elif key in RESERVED_KEYWORDS:
                continue

            else:
                # verify child options
                childOpts, childWarnings, childErrors = verify_options(options[key], autoCorrect)
                options[key] = childOpts
                warnings.extend(childWarnings)
                errors.extend(childErrors)
    
    return options, warnings, errors