/*
    Retrieves global project settings, such as the class definitions and the data query URI, from the server.

    2019 Benjamin Kellenberger
*/

window.loadConfiguration = function() {
    // general properties
    window.defaultColors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'];
    window.getDefaultColor = function(idx) {
        return window.defaultColors[idx % window.defaultColors.length];
    }

    // project-specific settings
    var promise = $.get('getProjectSettings', function(data) {
        window.dataServerURI = data['settings']['dataServerURI'];
        if(!window.dataServerURI.endsWith('/')) {
            window.dataServerURI += '/';
        }
        window.classes = data['settings']['classes'];
        window.annotationType = data['settings']['annotationType'];
        window.numImages_x = data['settings']['numImages_x'];
        window.numImages_y = data['settings']['numImages_y'];
        window.defaultImage_w = data['settings']['defaultImage_w'];
        window.defaultImage_h = data['settings']['defaultImage_h'];
    });

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
        },
        background: '#000000'
    };

    // labeling interface
    window.annotationProximityTolerance = 20;

    return promise;
};