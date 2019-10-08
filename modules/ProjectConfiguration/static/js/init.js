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
            console.log(data);
            var settings = data['settings'];

            // immutable settings
            window.annotationType = settings['annotationType'];
            window.predictionType = settings['predictionType'];

            // populate fields
            $('#field-project-name').val(settings['projectTitle']);
            $('#field-project-description').html(settings['projectDescr']);
            $('#public-checkbox').prop('checked', settings['isPublic']);
            $('#interface-enabled-checkbox').prop('checked', settings['interfaceEnabled']);
            $('#field-secret-token').val(window.location.href.replace('/configuration', '/enroll') + '?t=' + settings['secretToken']);
            $('#demo-checkbox').prop('checked', settings['demoMode']);
            $('#field-numImgsPerBatch').val(settings['numImagesPerBatch']);
            $('#field-minImageWidth').val(settings['minImageWidth']);
            $('#ai-model-enabled-checkbox').prop('checked', settings['aiModelEnabled']);

            
            // assemble class definitions
            //TODO



        }
    });


    // AI model metadata
    var aiModelSelect = $('#ai-model-class');
    var alModelSelect = $('#al-model-class');

    //TODO: flag for case when AIController is not available...
    var promise = $.ajax({
        url: '/getAvailableAImodels',
        method: 'GET',
        success: function(data) {
            // populate selection fields, if suitable for selected annotation and prediction types
            window.availableModels = data['models'];
            for(var key in data['models']['prediction']) {
                if(data['models']['prediction'][key]['annotationType'] === window.annotationType && 
                        data['models']['prediction'][key]['predictionType'] === window.predictionType) {
                    var entry = $('<option value="' + key + '">' + data['models']['prediction'][key]['name'] + '</option>');
                    aiModelSelect.append(entry);
                }
            }
            for(var key in data['models']['ranking']) {
                var entry = $('<option value="' + key + '">' + data['models']['ranking'][key]['name'] + '</option>');
                alModelSelect.append(entry);
            }
        },
        error: function(data) {
            //TODO
            console.log('ERROR:')
            console.log(data)
        }
    });

    promise = promise.done(function() {
        return $.ajax({
            url: 'getAImodelInfo',
            method: 'GET',
            success: function(data) {
                data = data['info'];

                // set selected AI and AL models
                aiModelSelect.val(data['ai_model_library']);
                $('#ai-model-class-descr').html(window.availableModels['prediction'][data['ai_model_library']]['description']);
                alModelSelect.val(data['ai_alcriterion_library']);
                $('#al-model-class-descr').html(window.availableModels['ranking'][data['ai_alcriterion_library']]['description']);
            }
        })
    });

    // show model descriptions upon change
    aiModelSelect.change(function() {
        var selModel = $(this).val();
        $('#ai-model-class-descr').html(window.availableModels['prediction'][selModel]['description']);
    });
    alModelSelect.change(function() {
        var selModel = $(this).val();
        $('#al-model-class-descr').html(window.availableModels['ranking'][selModel]['description']);
    });

    $('#ai-model-enabled-checkbox').change(function() {
        aiModelSelect.prop('disabled', !$(this).prop('checked'));
        alModelSelect.prop('disabled', !$(this).prop('checked'));
    });
});