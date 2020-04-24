/*
    Holds definitions of building blocks for designing
    workflows that provide basic markups and do parsing.

    2020 Benjamin Kellenberger
*/

class AbstractWorkflowTask {
    constructor() {

    }

    to_JSON() {
        throw Error('not implemented for abstract base class.')
    }

    get_markup() {
        throw Error('not implemented for abstract base class.')
    }
}


class TrainingWorkflowTask extends AbstractWorkflowTask {
    DEFAULT_VALUES = {
        //TODO
    }
    constructor() {

    }

    to_JSON() {

    }

    get_markup() {
        var markup = $('<div></div>');
        
        return markup;
    }
}