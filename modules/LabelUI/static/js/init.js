/*
    Sets up the frontend and loads all required parameters in correct order.

    2019 Benjamin Kellenberger
*/

$(document).ready(function() {

    // login check
    var promise = $.ajax({
        url: '/loginCheck',
        method: 'post',
        error: function() {
            window.location.href = '/';
        }
    });


    // set up general config
    promise = promise.done(function() {
        return window.loadConfiguration();
    });

    // command listener
    promise = promise.done(function() {
        window.commandListener = new CommandListener();
        return $.Deferred().promise();
    });

    // set up label class handler
    promise = promise.done(function() {
        window.labelClassHandler = new LabelClassHandler($('.legend-entries'));
        return $.Deferred().promise();
    });

    // set up data handler
    promise.done(function() {
        window.dataHandler = new DataHandler($('#gallery'));
        window.dataHandler.loadNextBatch();
    });

    // events
    window.eventTypes = [
        'keydown',
        'keyup',
        'mousein',
        'mouseout',
        'mousemove',
        'mousedown',
        'mouseup',
        'click',
        'wheel'
    ];

    // interface
    window.interfaceControls = {
        actions: {
            DO_NOTHING: 0,
            ADD_ANNOTATION: 1,
            REMOVE_ANNOTATIONS: 2
        }
    };
    window.interfaceControls.action = window.interfaceControls.actions.DO_NOTHING;
});