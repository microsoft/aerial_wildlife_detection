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
    //TODO: replaced with function in labelClassHandler.js
    // window.defaultColors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'];
    // window.getDefaultColor = function(idx) {
    //     return window.defaultColors[idx % window.defaultColors.length];
    // }

    // labeling interface
    window.annotationProximityTolerance = 15;
};



// project settings
//TODO: make proper promise object...
window.getProjectSettings = function() {
    return $.get('getProjectSettings', function(data) {
        window.projectName = data['settings']['projectName'];
        window.projectDescription = data['settings']['projectDescription'];
        window.dataServerURI = data['settings']['dataServerURI'];
        if(!window.dataServerURI.endsWith('/')) {
            window.dataServerURI += '/';
        }
        window.aiControllerURI = data['settings']['aiControllerURI'];
        if(window.aiControllerURI != null && !window.aiControllerURI.endsWith('/')) {
            window.aiControllerURI += '/';
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
        window.defaultBoxSize_w = parseInt(data['settings']['defaultBoxSize_w']);
        window.defaultBoxSize_h = parseInt(data['settings']['defaultBoxSize_h']);
        window.minBoxSize_w = parseInt(data['settings']['minBoxSize_w']);
        window.minBoxSize_h = parseInt(data['settings']['minBoxSize_h']);
        window.numImagesPerBatch = parseInt(data['settings']['numImagesPerBatch']);
        window.minImageWidth = parseInt(data['settings']['minImageWidth']);
        window.numImageColumns_max = parseInt(data['settings']['numImageColumns_max']);
        window.defaultImage_w = parseInt(data['settings']['defaultImage_w']);
        window.defaultImage_h = parseInt(data['settings']['defaultImage_h']);
        window.styles = data['settings']['styles'];


        // set interface page title and description
        if(window.projectName != null) {
            $('#project-title').html(window.projectName);
        }
        if(window.projectDescription != null) {
            $('#project-description').html(window.projectDescription);
        }


        // adjust number of images to one for mobile devices
        if (/Mobi|Android/i.test(navigator.userAgent)) {
            window.minImageWidth = '100%';
            window.numImageColumns_max = 1;
        }

        return $.Deferred().promise();
    });
}