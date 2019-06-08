/*
    Sets up the frontend and loads all required parameters in correct order.

    2019 Benjamin Kellenberger
*/

$(document).ready(function() {

    // set up general config
    var promise = window.loadConfiguration();

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
});