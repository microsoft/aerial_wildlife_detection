/*
    Sets up the frontend and loads all required parameters in correct order.

    2019 Benjamin Kellenberger
*/

$(document).ready(function() {

    // set up general config
    var promise = window.loadConfiguration();

    // styles (TODO: outsource?)
    window.styles = {
        hoverText: {
            offsetH: 10,
            box: {
                fill: 'rgba(88, 137, 216, 0.85)',
                stroke: {
                    color: '#FFFFFF',
                    lineWidth: 0.5
                },
                height: 16      // adjust according to font size
            },
            text: {
                font: '16px sans-serif bold',
                color: '#FFFFFF'
            }
        }
    };


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