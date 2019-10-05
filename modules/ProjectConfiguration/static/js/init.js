/**
 * Functionality to let admin users edit general project settings.
 * 
 * 2019 Benjamin Kellenberger
 */

$(document).ready(function() {
    
    // retrieve metadata
    $.ajax({
        url: 'getConfig',
        method: 'GET',
        success: function(data) {

            var settings = data['settings'];

            // populate fields
            $('#field-project-name').val(settings['projectTitle']);
            $('#field-project-description').html(settings['projectDescr']);
            $('#public-checkbox').prop('checked', settings['isPublic']);
            $('#field-secret-token').val(window.location.href.replace('/configuration', '/enroll') + '?t=' + settings['secretToken']);
            $('#demo-checkbox').prop('checked', settings['demoMode']);
            $('#field-numImgsPerBatch').val(settings['numImagesPerBatch']);
            $('#field-minImageWidth').val(settings['minImageWidth']);
            $('#ai-model-enabled-checkbox').val(settings['aiModelEnabled']);

            
            // assemble class definitions
            //TODO



        }
    });


});