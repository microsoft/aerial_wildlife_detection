/*
    Retrieves global project settings, such as the class definitions and the data query URI, from the server.

    2019 Benjamin Kellenberger
*/

window.parseBoolean = function(value) {
    return (value===1 || ['yes', '1', 'true'].includes(value.toString().toLowerCase()));
}

window.getCurrentDateString = function() {
    var date = new Date();
    return date.toString();
}

window.getRandomString = function() {
    // only used for temporary IDs, never for sensitive hashing
    return Math.random().toString(36).substring(7);
}

window.getRandomID = function() {
    return window.getCurrentDateString() + window.getRandomString();
}


window.loadConfiguration = function() {
    // general properties
    window.defaultColors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'];
    window.getDefaultColor = function(idx) {
        return window.defaultColors[idx % window.defaultColors.length];
    }

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
                height: 28      // adjust according to font size
            },
            text: {
                font: '14px sans-serif',
                color: '#FFFFFF'
            }
        },
        background: '#000000',
        resizeHandles: {
            size: 8,
            fillColor: '#FFFFFF',
            strokeColor: '#000000',
            lineWidth: 1
        },
        crosshairLines: {
            strokeColor: '#000000',
            lineWidth: 1,
            lineDash: [4, 4]
        }
    };

    // labeling interface
    window.annotationProximityTolerance = 5;
};



// project settings
//TODO: make proper promise object...
$.get('getProjectSettings', function(data) {
    window.projectName = data['settings']['projectName'];
    window.projectDescription = data['settings']['projectDescription'];
    window.dataServerURI = data['settings']['dataServerURI'];
    if(!window.dataServerURI.endsWith('/')) {
        window.dataServerURI += '/';
    }
    window.dataType = data['settings']['dataType'];
    window.minObjSize = data['settings']['minObjSize'];
    window.classes = data['settings']['classes'];
    window.enableEmptyClass = window.parseBoolean(data['settings']['enableEmptyClass']);
    window.annotationType = data['settings']['annotationType'];
    window.predictionType = data['settings']['predictionType'];
    window.showPredictions = window.parseBoolean(data['settings']['showPredictions']);
    window.showPredictions_minConf = data['settings']['showPredictions_minConf'];
    window.carryOverPredictions = window.parseBoolean(data['settings']['carryOverPredictions']);
    window.carryOverRule = data['settings']['carryOverRule'];
    window.carryOverPredictions_minConf = data['settings']['carryOverPredictions_minConf'];
    window.defaultBoxSize_w = data['settings']['defaultBoxSize_w'];
    window.defaultBoxSize_h = data['settings']['defaultBoxSize_h'];
    window.numImages_x = data['settings']['numImages_x'];
    window.numImages_y = data['settings']['numImages_y'];
    window.defaultImage_w = data['settings']['defaultImage_w'];
    window.defaultImage_h = data['settings']['defaultImage_h'];


    // set interface page title and description
    if(window.projectName != null) {
        $('#project-title').html(window.projectName);
    }
    if(window.projectDescription != null) {
        $('#project-description').html(window.projectDescription);
    }


    // adjust number of images to one for mobile devices
    if (/Mobi|Android/i.test(navigator.userAgent)) {
        window.numImages_x = 1;
        window.numImages_y = 1;
    }
});