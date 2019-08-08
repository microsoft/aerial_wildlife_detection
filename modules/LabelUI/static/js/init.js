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
        if(hex.toLowerCase().startsWith('rgb')) return hex;
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

    window.getBrightness = function(color) {
        var rgb = window.hexToRgb(color);
        match = /rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(,\s*\d+[\.\d+]*)*\)/g.exec(rgb);
        return (parseInt(match[1]) + parseInt(match[2]) + parseInt(match[3])) / 3;
    }
    

    
    window.shuffle = function(a) {
        var j, x, i;
        for (i = a.length - 1; i > 0; i--) {
            j = Math.floor(Math.random() * (i + 1));
            x = a[i];
            a[i] = a[j];
            a[j] = x;
        }
        return a;
    }


    // search function for label classes
    window.filterLabels = function() {
        var keywords = $('#labelclass-search-box').val();
        if(keywords != null && keywords != undefined) {
            keywords = keywords.split(/s[\s ]+/);
        }
        if(keywords.length === 0 || (keywords.length === 1 && keywords[0] === '')) keywords = null;
        window.labelClassHandler.filter(keywords);
    }

    // Levenshtein distance for word comparison
    window.levDist = function(s, t) {
        var d = []; //2d matrix
    
        // Step 1
        var n = s.length;
        var m = t.length;
    
        if (n == 0) return m;
        if (m == 0) return n;
    
        //Create an array of arrays in javascript (a descending loop is quicker)
        for (var i = n; i >= 0; i--) d[i] = [];
    
        // Step 2
        for (var i = n; i >= 0; i--) d[i][0] = i;
        for (var j = m; j >= 0; j--) d[0][j] = j;
    
        // Step 3
        for (var i = 1; i <= n; i++) {
            var s_i = s.charAt(i - 1);
    
            // Step 4
            for (var j = 1; j <= m; j++) {
    
                //Check the jagged ld total so far
                if (i == j && d[i][j] > 4) return n;
    
                var t_j = t.charAt(j - 1);
                var cost = (s_i == t_j) ? 0 : 1; // Step 5
    
                //Calculate the minimum
                var mi = d[i - 1][j] + 1;
                var b = d[i][j - 1] + 1;
                var c = d[i - 1][j - 1] + cost;
    
                if (b < mi) mi = b;
                if (c < mi) mi = c;
    
                d[i][j] = mi; // Step 6
    
                //Damerau transposition
                if (i > 1 && j > 1 && s_i == t.charAt(j - 2) && s.charAt(i - 2) == t_j) {
                    d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + cost);
                }
            }
        }
    
        // Step 7
        return d[n][m];
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

    // set up label class handler
    promise = promise.done(function() {
        window.labelClassHandler = new LabelClassHandler($('#legend-entries'));

        // search function
        $('#labelclass-search-box').on({
            keyup: window.filterLabels,
            search: window.filterLabels,
            focusin: function() {
                window.shortcutsDisabled = true;
                window.filterLabels();
            },
            focusout: function() {
                window.shortcutsDisabled = false;
                window.filterLabels();
            }
        });
    });

    // set up data handler
    promise = promise.then(function() {
        window.dataHandler = new DataHandler($('#gallery'));
    });


    // set up UI control
    promise = promise.then(function() {
        window.uiControlHandler = new UIControlHandler(window.dataHandler);
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
        window.shortcutsDisabled = false;       // if true, keystrokes like "A" for "label all" are disabled
        window.setUIblocked(true);


        // auto-resize entries on window resize
        window.windowResized = function () {
            var canvas = $('canvas');
            if(canvas.length === 0) return;
            canvas = canvas[0];

            var gallery = $('#gallery');
            var numCols = Math.min(Math.floor(gallery.width() / window.minImageWidth), window.numImageColumns_max);
            var numRows = Math.ceil(window.numImagesPerBatch / numCols);

            // resize canvas to fit height (so that as little scrolling as possible is needed)
            var aspectRatio = canvas.width / canvas.height;

            var height = Math.max(window.minImageWidth/aspectRatio, gallery.height() / numRows);
            var width = Math.max(window.minImageWidth, gallery.width() / numCols);
            if(height > width/aspectRatio) {
                height = width/aspectRatio;
            } else {
                width = height*aspectRatio;
            }

            var style = {
                'min-width':  window.minImageWidth+'px',
                'width': width + 'px',
                'height': height + 'px'
            };
            $('.entry').css(style);

            // update actual canvas width and height
            $('canvas').each(function() {
                this.width = width;
                this.height = height;
            })

            // gallery: account for width, center entries
            var toolsWidth = $('#tools-container').width() + $('#viewport-controls').width();
            $('#gallery').css({
                'width': 'calc(100vw - '+toolsWidth+'px)',
            });

            // label class entries
            $('#legend-entries').css('height', gallery.height() - 60 + 'px');   // -60 for search bar
            window.dataHandler.renderAll();

            // show or hide tooltip if scollable & adjust position
            $('#gallery-scroll-tooltip-bottom').css({
                'width': gallery.width(),
                'top': gallery.offset().top + gallery.outerHeight() - 25
            });
            window.galleryScrolled();
        }
        $(window).resize(windowResized);


        // show or hide tooltip in case of scrollable gallery
        window.galleryScrolled = function() {
            var gallery = $('#gallery');
            scrollPos = gallery.scrollTop();
            var galleryHeight = gallery[0].scrollHeight;
            var galleryHeight_visible = parseInt(gallery.height());

            // show tooltip if additional images invisible, hide if not
            if(galleryHeight - scrollPos - 1 > galleryHeight_visible) {
                $('#gallery-scroll-tooltip-bottom').fadeIn();
            } else {
                $('#gallery-scroll-tooltip-bottom').fadeOut();
            }
        }
        $('#gallery').on('scroll', window.galleryScrolled);
        $('#gallery-scroll-tooltip-bottom').click(function() {
            $('#gallery').animate({
                scrollTop: $('#gallery').height(),
                easing: 'swing'
            }, 500);
        });

        // // make tools container resizeable
        // $('#tools-container-resize-handle').on({
        //     mousedown: function(event) {
        //         this.currentX = event.pageX;
        //     },
        //     mousemove: function(event) {
        //         if(this.currentX != undefined && this.currentX != null) {
        //             var offset = this.currentX - event.pageX;
        //             var tc = $('#tools-container');
        //             tc.css('width', tc.css('width') + offset);
        //             windowResized();
        //         }
        //     },
        //     mouseup: function() {
        //         this.currentX = undefined;
        //     },
        //     mouseleave: function() {
        //         this.currentX = undefined;
        //     }
        // });


        // overlay HUD
        window.showOverlay = function(contents, large, uiBlocked_after) {
            if(contents === undefined || contents === null) {
                $('#overlay-card').slideUp(1000, function() {
                    $('#overlay-card').empty();

                    // reset style
                    $('#overlay-card').css('width', '720px');
                    $('#overlay-card').css('height', '250px');

                    if(!uiBlocked_after)
                        window.setUIblocked(false);
                });
                $('#overlay').fadeOut();

            } else {
                window.setUIblocked(true);

                // adjust style
                if(large) {
                    $('#overlay-card').css('width', '50%');
                    $('#overlay-card').css('height', '75%');
                }
                $('#overlay-card').html(contents);
                $('#overlay').fadeIn();
                $('#overlay-card').slideDown();
            }
        }


        // confirmation dialog
        window.showConfirmationDialog = function(requestID, callback_yes, callback_no, title, message, buttonText_yes, buttonText_no) {
            /*
                Shows a confirmation dialog with Yes/No buttons if, and only if, the requestID
                provided is defined and not found in the cookies. If the requestID is provided,
                the user is also presented with a checkbox that allows them to not show the
                message anymore, but skip directly to the confirmation callback the next time.
            */
            
            // go to callback directly if user requested not to show message anymore
            if(requestID != undefined && requestID != null) {
                var skip = window.getCookie(requestID);
                if(skip != undefined && skip != null) {
                    callback_yes();
                    return;
                }
            }

            // create markup
            if(title === undefined || title === null) title = 'Confirm';
            if(message === undefined || message === null) message = 'Would you like to proceed?';
            if(buttonText_yes === undefined || buttonText_yes === null) buttonText_yes = 'Yes';
            if(buttonText_no === undefined || buttonText_no === null) buttonText_no = 'No';

            var cookieCheckbox = $('<input type="checkbox" style="margin-right:10px" />');
            var cookieLabel = $('<label style="margin-top:20px">Do not show this message again.</label>').prepend(cookieCheckbox);
            var button_yes = $('<button class="btn btn-primary btn-sm" style="display:inline;float:right">' + buttonText_yes + '</button>');
            var button_no = $('<button class="btn btn-secondary btn-sm" style="display:inline">' + buttonText_no + '</button>');
            var buttons = $('<div></div>').append(button_no).append(button_yes);
            var markup = $('<div><h2>' + title + '</h2><div>' + message + '</div></div>').append(cookieLabel).append(buttons);

            // wrap callbacks
            var dispose = function() {
                // check if cookie is to be set
                var skipMsg = cookieCheckbox.is(':checked');
                if(skipMsg) {
                    window.setCookie(requestID, true, 365);
                }
                window.showOverlay(null);
            }
            var action_yes = function() {
                dispose();
                callback_yes();
            }
            var action_no = function() {
                dispose();
                if(callback_no != undefined && callback_no != null) {
                    callback_no();
                }
            }
            button_yes.click(action_yes);
            button_no.click(action_no);

            window.showOverlay(markup, false, false);
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