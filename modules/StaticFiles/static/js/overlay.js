/**
 * Tools for showing a general overlay over the UI, e.g.
 * for session renewal.
 * 
 * 2020 Benjamin Kellenberger
 */


// General overlay functionality
$(document).ready(function() {
    window.overlay = $('#overlay-card');
    window.overlayAvailable = (window.overlay.length > 0);
    window.uiBlockAvailable = window.hasOwnProperty('setUIblocked');
});

window.showOverlay = function(contents, large, uiBlocked_after) {
    if(!window.overlayAvailable) return;
    if(contents === undefined || contents === null) {
        window.overlay.slideUp(1000, function() {
            window.overlay.empty();

            // reset style
            window.overlay.css('width', '720px');
            window.overlay.css('height', '250px');

            if(window.uiBlockAvailable && !uiBlocked_after)
                window.setUIblocked(false);
        });
        $('#overlay').fadeOut();

    } else {
        if(window.uiBlockAvailable)
            window.setUIblocked(true);

        // adjust style
        if(large) {
            window.overlay.css('width', '50%');
            window.overlay.css('height', '75%');
        }
        window.overlay.html(contents);
        $('#overlay').fadeIn();
        window.overlay.slideDown();
    }
}


// Login verification / session renewal
window.renewSessionRequest = function(xhr) {
    /**
     * Function can be called with a failed AJAX request
     * ("error"); parses the error for 401 "unauthorized"
     * responses. If the error happens to be due to authori-
     * zation, the session renewal overlay is shown and a
     * promise is returned that awaits the user's password
     * input. Otherwise, a deferred promise is returned.
     */
    if(typeof(xhr) === 'object' && xhr.hasOwnProperty('status') && xhr['status'] === 401) {
        return window.verifyLogin();
    } else {
        return $.Deferred().promise();
    }
}

window.verifyLogin = function(callback) {
    return $.ajax({
        url: 'loginCheck',
        method: 'post',
        success: function() {
            window.showOverlay(null);
            if(typeof(callback) === 'function')
                callback();
        },
        error: function() {
            // show login verification overlay
            window.showVerificationOverlay(callback);
        }
    });
}

window.showVerificationOverlay = function(callback) {
    if(!window.overlayAvailable) return;
    var loginFun = function(callback) {
        var username = $('#navbar-user-dropdown').html();       // cannot use cookie since it has already been deleted by the server
        var password = $('#password').val();
        $.ajax({
            url: 'doLogin',
            method: 'post',
            data: {username: username, password: password},
            success: function(response) {
                window.showOverlay(null);
                if(typeof(callback) === 'function')
                    callback();
            },
            error: function(error) {
                $('#invalid-password').show();
            }
        })
    }

    var overlayHtml = $('<h2>Renew Session</h2><div class="row fieldRow">' +
                        '<label for="password" class="col-sm">Password:</label>' +
                        '<input type="password" name="password" id="password" required class="col-sm" /></div>' +
                        '<div class="row fieldRow"><div class="col-sm">'+
                        '<div id="invalid-password" style="display:none;color:red;">invalid password entered</div>' +
                        '<button id="abort" class="btn btn-sm btn-danger">Cancel</button>' +
                        '<button id="confirm-password" class="btn btn-sm btn-primary float-right">OK</button></div></div>');
    window.showOverlay(overlayHtml, false, false);

    $('#abort').click(function() {
        window.location.href = '/';     //TODO: '/login'
    })

    $('#confirm-password').click(function() {
        loginFun(callback);
    });
}