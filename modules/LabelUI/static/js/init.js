/*
    Sets up the frontend and loads all required parameters in correct order.

    2019 Benjamin Kellenberger
*/

$(document).ready(function() {

    // enable/disable interface
    window.setUIblocked = function(blocked) {
        window.uiBlocked = blocked;
        $('button').prop('disabled', blocked);
    }

    // loading overlay
    window.showLoadingOverlay = function(visible) {
        if(visible) {
            window.setUIblocked(true);
            $('#overlay').css('display', 'block');
            $('#overlay-loader').css('display', 'block');
            $('#overlay-card').css('display', 'none');

        } else {
            $('#overlay').fadeOut({
                complete: function() {
                    $('#overlay-loader').css('display', 'none');
                }
            });
            window.setUIblocked(false);
        }
    }


    // block UI until loaded
    window.showLoadingOverlay(true);



    // cookie helper
    window.getCookie = function(name) {
        var match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
        if (match) return match[2];
    }
    window.setCookie = function(name, value, days) {
        var d = new Date;
        d.setTime(d.getTime() + 24*60*60*1000*days);
        document.cookie = name + "=" + value + ";path=/;expires=" + d.toGMTString();
    }


    // time util
    window.msToTime = function(duration) {
        var seconds = Math.floor((duration / 1000) % 60),
            minutes = Math.floor((duration / (1000 * 60)) % 60),
            hours = Math.floor((duration / (1000 * 60 * 60)) % 24);

        if(hours > 0) {
            hours = (hours < 10) ? '0' + hours : hours;
            minutes = (minutes < 10) ? '0' + minutes : minutes;
            seconds = (seconds < 10) ? '0' + seconds : seconds;
            result = hours + ':' + minutes + ':' + seconds;
            return result;

        } else {
            minutes = (minutes < 10) ? '0' + minutes : minutes;
            seconds = (seconds < 10) ? '0' + seconds : seconds;
            return minutes + ':' + seconds;
        }
    }


    // color converter: adds alpha channel to existing color string
    window.hexToRgb = function(hex) {
        // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
        var shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
        hex = hex.replace(shorthandRegex, function(m, r, g, b) {
            return r + r + g + g + b + b;
        });

        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? 'rgb(' + 
                parseInt(result[1], 16) + ',' + 
                parseInt(result[2], 16) + ',' + 
                parseInt(result[3], 16) + ')' : null;
    }

    window._addAlpha = function(color, alpha) {
        a = alpha > 1 ? (alpha / 100) : alpha;
        if(color.startsWith('#')) {
            // HEX color string
            color = window.hexToRgb(color);
        }
        match = /rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(,\s*\d+[\.\d+]*)*\)/g.exec(color);
        return "rgba(" + [match[1],match[2],match[3],a].join(',') +")";
    }

    window.addAlpha = function(color, alpha) {
        if(color === null || color === undefined) return null;
        if(alpha === null || alpha === undefined) return color;
        if(alpha <= 0.0) return null;
        alpha = alpha > 1 ? (alpha / 100) : alpha;
        if(alpha >= 1.0) return color;
        return window._addAlpha(color, alpha);
    }
    


    // login check
    var promise = $.ajax({
        url: '/loginCheck',
        method: 'post',
        error: function() {
            window.location.href = '/';
        }
    });

    // set up general config
    promise = promise.then(function() {
        return window.loadConfiguration();
    });

    promise = promise.then(function() {
        return window.getProjectSettings();
    });

    // command listener
    promise = promise.then(function() {
        window.commandListener = new CommandListener();
    });

    // set up label class handler
    promise = promise.done(function() {
        window.labelClassHandler = new LabelClassHandler($('.legend-entries'));
    });

    // set up data handler
    promise = promise.then(function() {
        window.dataHandler = new DataHandler($('#gallery'));
    });


    promise = promise.then(function() {
        // events
        window.eventTypes = [
            'keydown',
            'keyup',
            'mousein',
            'mouseout',
            'mouseleave',
            'mousemove',
            'mousedown',
            'mouseup',
            'click',
            'wheel'
        ];

        // interface
        window.interfaceControls = {
            actions: {
                DO_NOTHING: 0,
                ADD_ANNOTATION: 1,
                REMOVE_ANNOTATIONS: 2
            }
        };
        window.interfaceControls.action = window.interfaceControls.actions.DO_NOTHING;

        window.setUIblocked(true);


        // make class panel grow and shrink on mouseover/mouseleave
        $('#tools-container').on('mouseenter', function() {
            if(window.uiBlocked || $(this).is(':animated')) return;
            $('#tools-container').animate({
                right: 0
            });
        });
        $('#tools-container').on('mouseleave', function() {
            if(window.uiBlocked) return;
            let offset = -$(this).outerWidth() + 40;
            $('#tools-container').animate({
                right: offset
            });
        });
        $('#tools-container').css('right', -$('#tools-container').outerWidth() + 40);


        // overlay HUD
        window.showOverlay = function(contents) {
            if(contents === undefined || contents === null) {
                $('#overlay-card').slideUp();
                $('#overlay').fadeOut();
                $('#overlay-card').empty();
                window.setUIblocked(false);

            } else {
                window.setUIblocked(true);
                $('#overlay-card').html(contents);
                $('#overlay').fadeIn();
                $('#overlay-card').slideDown();
            }
        }


        // login verification screen
        window.showVerificationOverlay = function(callback) {
            var loginFun = function(callback) {
                var username = $('#navbar-user-dropdown').html();       // cannot use cookie since it has already been deleted by the server
                var password = $('#password').val();
                $.ajax({
                    url: '/login',
                    method: 'post',
                    data: {username: username, password: password},
                    success: function(response) {
                        window.showOverlay(null);
                        callback();
                    },
                    error: function(error) {
                        $('#invalid-password').show();
                    }
                })
            }

            var overlayHtml = $('<h2>Renew Session</h2><div class="row fieldRow"><label for="password" class="col-sm">Password:</label><input type="password" name="password" id="password" required class="col-sm" /></div><div class="row fieldRow"><div class="col-sm"><div id="invalid-password" style="display:none;color:red;">invalid password entered</div><button id="abort" class="btn btn-sm btn-danger">Cancel</button><button id="confirm-password" class="btn btn-sm btn-primary float-right">OK</button></div></div>');
            window.showOverlay(overlayHtml);

            $('#abort').click(function() {
                window.location.href = '/';
            })

            $('#confirm-password').click(function() {
                loginFun(callback);
            });
        }

        window.verifyLogin = function(callback) {
            return $.ajax({
                url: '/loginCheck',
                method: 'post',
                success: function() {
                    window.showOverlay(null);
                    callback();
                },
                error: function() {
                    // show login verification overlay
                    window.showVerificationOverlay(callback);
                }
            });
        }



        // logout and reload functionality
        window.onbeforeunload = function() {
            window.dataHandler.submitAnnotations(true);
        };

        $('#logout').click(function() {
            window.dataHandler.submitAnnotations(true);
            window.location.href = '/logout';
        });
    });


    // AI backend
    promise = promise.then(function() {
        if(window.aiControllerURI != null) {
            window.aiWorkerHandler = new AIWorkerHandler($('.ai-worker-entries'));
            $('#ai-worker-minipanel').show();
        }
    });


    // load image batch
    promise = promise.then(function() {
        return window.dataHandler._loadNextBatch();
    });


    // enable interface
    promise = promise.then(function() {
        window.showLoadingOverlay(false);
    });


    // show interface tutorial
    promise.then(function() {
        if(!(window.getCookie('skipTutorial')))
            window.showTutorial(true);
    });
});