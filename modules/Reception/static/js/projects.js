let projects = {};      // general info about projects
let selectedFolder = 'all-projects';


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
                    let projDescr = data['projects'][key]['description'];
                    let archived = data['projects'][key]['archived'];
                    let demoMode = data['projects'][key]['demoMode'];
                    let isOwner = data['projects'][key]['isOwner'];
                    
                    let role = data['projects'][key]['role'];

                    if(archived) {
                        // append to separate table
                        let markup = $('<tr id="archivedEntry_'+key+'"></tr>');
                        markup.append($('<td><a href="' + key + '">' + projName + '</a></td>'));

                        if(isOwner || role === 'super user') {
                            // only owners and super users are allowed to unarchive a project
                            let unarchive = $('<td></td>');
                            let unarchiveBtn = $('<button class="btn btn-sm btn-primary unarchive-button">Unarchive</button>');
                            unarchiveBtn.on('click', function() {
                                //TODO
                                window.messager.addMessage('Button not yet implemented. Please unarchive project <a href="' + key + '/configuration/dangerZone">here</a>.', 'error', 0);
                            });
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
                            adminButtons = '<span class="project-buttons"><a href="' + key + '/configuration/verview" class="btn btn-sm btn-success">Statistics</a>' +
                                        '<a href="' + key + '/configuration/general" class="btn btn-sm btn-secondary">Configure</a>';
                            if(data['projects'][key]['aiModelEnabled']) {
                                adminButtons += '<a href="' + key + '/configuration/aiModel" class="btn btn-sm btn-info">AI model</a>';
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
                        markup.append($('<p>' + projDescr + '</p>'));
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
                        meta: {
                            name: projName,
                            description: projDescr,
                        },
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


function _string_matches(object, keywords) {
    if(object === undefined || object === null) {
        return false;
    } else if(Array.isArray(object)) {
        for(var k=0; k<object.length; k++) {
            if(_string_matches(object[k], keywords)) return true;
        }
        return false;
    } else if(typeof(object) === 'object') {
        let keys = Object.keys(object);
        for(var k=0; k<keys.length; k++) {
            if(_string_matches(object[keys[k]], keywords))
                return true;
        }
        return false;
    } else {
        object = object.toString().toLowerCase();
        for(var k=0; k<keywords.length; k++) {
            if(object.includes(keywords[k].toLowerCase()))
                return true;
        }
        return false;
    }
}


function containsKeywords(object, searchString) {
    if(searchString === undefined || searchString === null || searchString.length === 0) return true;
    let keywords = searchString.toLowerCase().split(' ');
    if(keywords.length === 0) return true;
    let hasValidKeyword = false;
    for(var k=0; k<keywords.length; k++) {
        keywords[k] = keywords[k].trim();
        if(keywords[k].length > 0) hasValidKeyword = true;
    }
    if(!hasValidKeyword) return true;
    return _string_matches(object, keywords);
}


function filterProjects(folderType, keywords) {
    if(folderType === 'projects-archived') {
        $('#main-projects-panel').hide();
        $('#archived-projects-panel').show();

        // check keywords
        $('#projects-archived-tbody').children().each(function() {
            let id = $(this).attr('id').replace('archivedEntry_', '');
            if(containsKeywords(projects[id]['meta'], keywords)) {
                $(this).show();
            } else {
                $(this).hide();
            }
        });

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
            
            // check keywords
            let keywordMatch = containsKeywords(projects[key]['meta'], keywords);

            if((attr === undefined || projects[key][attr]) && keywordMatch) {
                // all projects or attributes as well as keywords match
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
            selectedFolder = $(this).attr('id');
            filterProjects(selectedFolder, $('#project-search-field').val());
        });
    });

    // search field
    $('#project-search-field').on('input', function() {
        filterProjects(selectedFolder, $('#project-search-field').val());
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