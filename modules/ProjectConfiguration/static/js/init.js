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

            // immutable settings
            window.annotationType = settings['annotationType'];
            window.predictionType = settings['predictionType'];

            // populate fields
            $('#field-project-name').val(settings['projectTitle']);
            $('#field-project-description').html(settings['projectDescr']);
            $('#public-checkbox').prop('checked', settings['isPublic']);
            $('#interface-enabled-checkbox').prop('checked', settings['interfaceEnabled']);
            $('#field-secret-token').val(window.location.href.replace('/configuration', '/enroll') + '?t=' + settings['secretToken']);
            window.secretToken = settings['secretToken'];
            $('#demo-checkbox').prop('checked', settings['demoMode']);
            $('#field-numImgsPerBatch').val(settings['numImagesPerBatch']);
            $('#field-minImageWidth').val(settings['minImageWidth']);
        }
    });


    // // Label class definitions
    // $.ajax({
    //     url: 'getClassDefinitions',
    //     method: 'GET',
    //     success: function(response) {
    //         // parse and re-structure class definitions (TODO: do on server?)
    //         //TODO: keystroke?
    //         response = response['classes']['entries'];
    //         data = [];

    //         var _parse_group = function(group) {
    //             group['text'] = group['name'];
    //             if(group.hasOwnProperty('color') && group['color'] != null) {
    //                 group['backColor'] = group['color'];
    //             } else {
    //                 group['backColor'] = '#000';
    //             }
    //             group['selectable'] = true;
    //             group['icon'] = 'glyphicon glyphicon-glyphicon-menu-right';
    //             if(group.hasOwnProperty('entries')) {
    //                 var entries = [];
    //                 for(var key in group['entries']) {
    //                     entries.push(_parse_group(group['entries'][key]));
    //                 }
    //                 group['nodes'] = entries;
    //             }
    //             return group;
    //         }

    //         for(var key in response) {
    //             var nextGroup = _parse_group(response[key]);
    //             data.push(nextGroup);
    //         }
            
    //         $('#label-class-tree').treeview({data: data});
    //     },
    //     error: function(data) {
    //         console.log(data);
    //     }
    // });

    // // label class buttons
    // $('#remove-lc-button').click(function() {
    //     var lcData = $('#label-class-tree');
    //     console.log(lcData.treeview('getSelected'));
    // })



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
            url: 'getAImodelSettings',
            method: 'GET',
            success: function(data) {
                data = data['settings'];

                // checkbox
                $('#ai-model-enabled-checkbox').prop('checked', data['aiModelEnabled']);

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


    // main buttons
    $('#proj-settings-save-button').click(function() {

        // assemble settings
        var settings = {};

        settings['projectDescr'] = $('#field-project-description').html();
        settings['isPublic'] = $('#public-checkbox').prop('checked');
        settings['interfaceEnabled'] = $('#interface-enabled-checkbox').prop('checked');
        settings['secretToken'] = window.secretToken;
        settings['demoMode'] = $('#demo-checkbox').prop('checked');

        // UI
        settings['ui_settings'] = {
            'numImagesPerBatch': $('#field-numImgsPerBatch').val(),
            'minImageWidth': $('#field-minImageWidth').val()
        };

        
        //TODO: separate ajax call to AIController
        // // AI
        // settings['aiModelEnabled'] = $('#ai-model-enabled-checkbox').prop('checked');
        // settings['ai_model_library'] = $('#ai-model-class').val();
        // settings['al_model_library'] = $('#al-model-class').val();

        
        // general project settings
        $.ajax({
            url: 'saveProjectConfiguration',
            method: 'POST',
            data: JSON.stringify(settings),
            contentType: 'application/json; charset=utf-8',
            dataType: 'json',
            success: function(data) {
                console.log(data);
            },
            error: function(data) {
                console.log(data);
            }
        });


        // AI settings
        //TODO
    });
});