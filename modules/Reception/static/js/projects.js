let projects = {};      // general info about projects


function loadProjectInfo() {
    let projDiv = $('#projects');
    let projArchivedDiv = $('#projects-archived-tbody');
    return $.ajax({
        url: 'getProjects',
        method: 'GET',
        success: function(data) {
            if(data.hasOwnProperty('projects')) {
                projects = {};
                for(var key in data['projects']) {
                    let projName = data['projects'][key]['name'];
                    let archived = data['projects'][key]['archived'];
                    let demoMode = data['projects'][key]['demoMode'];
                    let isOwner = data['projects'][key]['isOwner'];
                    
                    let role = data['projects'][key]['role'];

                    if(archived) {
                        // append to separate table
                        let markup = $('<tr></tr>');
                        markup.append($('<td><a href="' + key + '">' + projName + '</a></td>'));

                        if(isOwner || role === 'super user') {
                            // only owners and super users are allowed to unarchive a project
                            let unarchive = $('<td></td>');
                            let unarchiveBtn = $('<button class="btn btn-sm btn-primary">Unarchive</button>');
                            unarchiveBtn.on('click', function() {
                                console.log('I would now unarchive this project: ' + key);
                            })
                            unarchive.append(unarchiveBtn);
                            markup.append(unarchive);
                        } else {
                            markup.append($('<td></td>'));
                        }
                        projArchivedDiv.append(markup);

                    } else {
                        if(role === 'super user' || role === 'admin' || role === 'member') {
                        } else {
                            role = 'not a member';
                        }
                        let userAdmitted = data['projects'][key]['userAdmitted'];
                        let adminButtons = '';
                        if(role === 'admin' || role === 'super user') {
                            // show button to project configuration page
                            adminButtons = '<span class="project-buttons"><a href="' + key + '/configuration?t=overview" class="btn btn-sm btn-success">Statistics</a>' +
                                        '<a href="' + key + '/configuration?t=general" class="btn btn-sm btn-secondary">Configure</a>';
                            if(data['projects'][key]['aiModelEnabled']) {
                                adminButtons += '<a href="' + key + '/configuration?t=aiModel" class="btn btn-sm btn-info">AI model</a>';
                            }
                            //TODO: implement correctly:
                            // if(isOwner) {
                            //     let archiveButton = '<button class="btn btn-sm btn-warning">Archive</button>';
                            //     $(archiveButton).on('click', function() {
                            //         //TODO
                            //         console.log(key);
                            //     })
                            //     adminButtons += archiveButton;
                            // }
                            adminButtons += '</span>';
                            userAdmitted = true;
                            var authDescr = $('<p style="display:inline">You are <b>' + role + '</b> in this project.</p>');
                        } else if(data['projects'][key]['demoMode']) {
                            var authDescr = $('<p style="display:inline">You are allowed to view (but not label) the images in this project.</p>');
                        }
                        
                        if(demoMode) {
                            labelButtonText = 'Explore';
                        } else {
                            labelButtonText = 'Start labeling';
                        }

                        var labelButton = '<a href="' + key + '/interface" class="btn btn-primary label-button">'+labelButtonText+'</a>';
                        if(!userAdmitted || !data['projects'][key]['interface_enabled']) {
                            labelButton = '<div class="btn btn-secondary label-button" style="cursor:not-allowed;" disabled="disabled">(interface disabled)</div>';
                        }

                        var markup = $('<div class="project-entry" id="projectEntry_' + key + '"></div>');
                        markup.append($('<h2><a href="' + key + '">' + projName + '</a></h2>'));
                        markup.append($('<p>' + data['projects'][key]['description'] + '</p>'));
                        markup.append(authDescr);
                        if(demoMode) {
                            markup.append($('<p>Project is in demo mode.</p>'));
                        }
                        markup.append('<div style="height:20px">' + labelButton +
                        adminButtons +
                            '</div>');
                        projDiv.append(markup);
                    }

                    projects[key] = {
                        demoMode: demoMode,
                        isOwner: isOwner,
                        isAdmin: (role === 'admin' || isOwner),        //TODO: super user
                        sharedWithYou: (role === 'member'),
                        archived: archived
                    }
                }

            }
        },
        error: function(data) {
            console.log(data);
        }
    });
}


function filterProjects(folderType) {
    if(folderType === 'projects-archived') {
        $('#main-projects-panel').hide();
        $('#archived-projects-panel').show();

    } else {
        let attr = undefined
        switch(folderType) {
            case 'your-projects':
                attr = 'isOwner';
                break;
            case 'admin-projects':
                attr = 'isAdmin';
                break;
            case 'projects-shared':
                attr = 'sharedWithYou';
                break;
            case 'projects-demo':
                attr = 'demoMode';
                break;
        }
        for(var key in projects) {
            let markup = $('#projectEntry_'+key);
            if(attr === undefined || projects[key][attr]) {
                // all projects or attribute matches
                markup.show();
            } else {
                markup.hide();
            }
        }

        $('#main-projects-panel').show();
        $('#archived-projects-panel').hide();
    }

    $('#project-folders').children().each(function() {
        if($(this).attr('id') === folderType) {
            $(this).addClass('active');
        } else {
            $(this).removeClass('active');
        }
    });
}


$(document).ready(function() {

    // navigation controls
    $('#project-folders').children().each(function() {
        $(this).on('click', function() {
            filterProjects($(this).attr('id'));
        });
    });


    // hide name ta (log in button) if not logged in (out)
    if($('#navbar-user-dropdown').html() === '') {
        $('#navbar-dropdown').hide();
    } else {
        $('#login-button').hide();
    }


    loadProjectInfo();


    // check if logged in and show placeholder if not
    $.ajax({
        url: 'loginCheck',
        method: 'post',
        success: function() {
            $('#projects-placeholder').hide();

            // show navigation sidebar
            $('#navigation-panel').show();
            $('#side-border').show();   //TODO: ugly hack
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
    });


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
    });
});