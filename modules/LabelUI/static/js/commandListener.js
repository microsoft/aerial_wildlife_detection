/*
    Sets different variables according to user interaction
    (e.g. key press, button press, etc.).

    2019 Benjamin Kellenberger
*/

const interactionTypes = {
    HOVER: 'hover',
    CLICK: 'click',
    KEYDOWN: 'keydown',
    KEYUP: 'keyup',
    KEYPRESS: 'keypress'
};

const interfaceCommands = {
    DO_NOTHING: null,
    ZOOM_IN: 'zoomIn',
    ZOOM_OUT: 'zoomOut',
    ZOOM_RECT: 'zoomRect',
    ADD_ANNOTATION: 'addAnnotation'
};


class CommandListener {

    constructor() {
        this.registeredEvents = {};
    }

    _addGenericCallback(element, type, callbackFun) {
        $(element).on(type, function(event) {
            callbackFun(event);
        });
    }

    _addKeyCallback(element, type, key, callbackFun) {
        //TODO: implement window-wide keydowns?
        $(element).on(type, function(event) {
            if(event.which() == key) {
                callbackFun(event);
            }
        });
    }

    addCallback(element, type, callbackFun, args) {
        if(type.includes('key')) {
            this._addKeyCallback(element, type, args, callbackFun);
            this.registeredEvents[element] = type;
        } else {
            this._addGenericCallback(element, type, callbackFun);
            this.registeredEvents[element] = type;
        }
    }

    removeCallback(element) {
        if(element in this.registeredEvents) {
            $(element).off(this.registeredEvents[element]);
            delete this.registeredEvents[element];
        }
    }
}