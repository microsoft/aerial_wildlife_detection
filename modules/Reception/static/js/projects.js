$(document).ready(function() {

    // hide name ta (log in button) if not logged in (out)
    if($('#navbar-user-dropdown').html() === '') {
        $('#navbar-dropdown').hide();
    } else {
        $('#login-button').hide();
    }

    // load project info
    var projDiv = $('#projects');
    $.ajax({
        url: 'getProjects',
        method: 'GET',
        success: function(data) {
            if(data.hasOwnProperty('projects')) {
                for(var key in data['projects']) {
                    var role = data['projects'][key]['role'];
                    if(role === 'super user' || role === 'admin' || role === 'member') {
                        var isMember = true;
                    } else {
                        role = 'not a member';
                        var isMember = false;
                    }
                    var userAdmitted = data['projects'][key]['userAdmitted'];
                    var adminButton = '';
                    if(role === 'admin' || role === 'super user') {
                        // show button to project configuration page
                        adminButton = '<a href="' + key + '/configuration" class="btn btn-secondary">Configure</a>';
                        userAdmitted = true;
                        var authDescr = $('<p>You are <b>' + role + '</b> in this project.</p>');
                    } else if(data['projects'][key]['demoMode']) {
                        var authDescr = $('<p>You are allowed to view (but not label) the images in this project.</p>');
                    }

                    var labelButton = '<a href="' + key + '/interface" class="btn btn-primary label-button">Start labeling</a>';
                    if(!userAdmitted || !data['projects'][key]['interfaceEnabled']) {
                        labelButton = '<div class="btn btn-secondary label-button" style="cursor:not-allowed;" disabled="disabled">(interface disabled)</div>';
                    }

                    var markup = $('<div class="project-entry"></div>');
                    markup.append($('<h2><a href="' + key + '">' + data['projects'][key]['name'] + '</a></h2>'));
                    markup.append($('<p>' + data['projects'][key]['description'] + '</p>'));
                    markup.append(authDescr);
                    markup.append('<div>' + labelButton +
                        adminButton +
                        '</div></div>');
                    projDiv.append(markup);
                }

            }
        },
        error: function(data) {
            console.log(data);
        }
    });


    // check if logged in and show placeholder if not
    $.ajax({
        url: 'loginCheck',
        method: 'post',
        success: function() {
            $('#projects-placeholder').hide();
        },
        error: function() {
            $('#projects-placeholder').show();
        }
    });


    // show "create account" button if not restricted
    $.ajax({
        url: 'getCreateAccountUnrestricted',
        method: 'GET',
        success: function(data) {
            if(data.hasOwnProperty('response') && data['response'] === true) {
                $('#create-account-panel').show();
            } else {
                $('#create-account-panel').hide();
            }
        },
        error: function() {
            $('#create-account-panel').hide();
        }
    })


    // show "new project" button if user is authorized to do so
    $.ajax({
        url: 'getAuthentication',
        method: 'GET',
        success: function(data) {
            if(data.hasOwnProperty('authentication')) {
                if(data['authentication']['canCreateProjects'] || data['authentication']['isSuperUser']) {
                    $('#new-project-button').show();
                }
            }
        }
    })
});