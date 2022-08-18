/**
 * Tools for showing a general overlay over the UI, e.g.
 * for session renewal.
 * 
 * 2020-22 Benjamin Kellenberger
 */


// General overlay functionality
$(document).ready(function() {
    window.overlay = $('#overlay-card');
    window.overlayAvailable = (window.overlay.length > 0);
    window.uiBlockAvailable = window.hasOwnProperty('setUIblocked');

    if(window.baseURL === undefined) {
        window.baseURL = '';
    }
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


// Yes-No screen
window.showYesNoOverlay = function(contents, callbackYes, callbackNo, buttonTextYes, buttonTextNo, buttonClassesYes, buttonClassesNo, large, uiBlocked_after, keepOverlayOnYes) {
    let markup = $('<div></div>');
    if(typeof(buttonTextYes) !== 'string' || buttonTextYes.length === 0) {
        buttonTextYes = 'Yes';
    }
    if(typeof(buttonTextNo) !== 'string' || buttonTextNo.length === 0) {
        buttonTextNo = 'Yes';
    }
    let bclYes = buttonClassesYes;
    if(typeof(bclYes) !== 'string' || bclYes.length === 0) {
        bclYes = 'btn btn-primary';
    }
    if(bclYes.indexOf('/\bbtn\b/') === -1) {
        bclYes = 'btn ' + bclYes;
    }
    let bclNo = buttonClassesNo;
    if(typeof(bclNo) !== 'string' || bclNo.length === 0) {
        bclNo = 'btn btn-secondary';
    }
    if(bclNo.indexOf('/\bbtn\b/') === -1) {
        bclNo = 'btn ' + bclNo;
    }
    let buttonYes = $('<button style="float:right" class="'+bclYes+'">'+buttonTextYes+'</button>');
    buttonYes.on('click', function() {
        if(!keepOverlayOnYes) window.showOverlay(null);
        if(typeof(callbackYes) === 'function') {
            callbackYes();
        }
    });
    let buttonNo = $('<button class="'+bclNo+'">'+buttonTextNo+'</button>');
    buttonNo.on('click', function() {
        window.showOverlay(null);
        if(typeof(callbackNo) === 'function') {
            callbackNo();
        }
    });
    let buttonMarkup = $('<div style="margin-top:10px"></div>');
    buttonMarkup.append(buttonNo);
    buttonMarkup.append(buttonYes);
    if(typeof(contents) !== 'undefined') {
        markup.append($(contents));
    }
    markup.append(buttonMarkup);
    window.showOverlay(markup, large, uiBlocked_after);
}


// Login verification / session renewal
window.renewSessionRequest = function(xhr, callback) {
    /**
     * Function can be called with a failed AJAX request
     * ("error"); parses the error for 401 "unauthorized"
     * responses. If the error happens to be due to authori-
     * zation, the session renewal overlay is shown and a
     * promise is returned that awaits the user's password
     * input. Otherwise, a deferred promise is returned.
     */
    if(typeof(xhr) === 'object' && xhr.hasOwnProperty('status') && xhr['status'] === 401) {
        return window.verifyLogin(callback);
    } else {
        return $.Deferred().promise();
    }
}

window.verifyLogin = function(callback) {
    return $.ajax({
        url: window.baseURL + 'loginCheck',
        method: 'post',
        success: function() {
            window.showOverlay(null);
            if(typeof(callback) === 'function')
                return callback();
        },
        error: function() {
            // show login verification overlay
            return window.showVerificationOverlay(callback);
        }
    });
}

window.showVerificationOverlay = function(callback) {
    if(!window.overlayAvailable) return $.Deferred().promise();
    var loginFun = function(callback) {
        var username = $('#navbar-user-dropdown').html();       // cannot use cookie since it has already been deleted by the server
        var password = $('#password').val();
        return $.ajax({
            url: window.baseURL + 'doLogin',
            method: 'post',
            data: {username: username, password: password},
            success: function(response) {
                window.showOverlay(null);
                if(typeof(callback) === 'function')
                    return callback();
                else
                    return $.Deferred().promise();
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
        window.location.href = '/';
    })

    $('#confirm-password').click(function() {
        loginFun(callback);
    });
}