/*
    Sets up the frontend and loads all required parameters in correct order.

    2019-20 Benjamin Kellenberger
*/

$(document).ready(function() {

    // hide user menu if not logged in
    if($('#navbar-user-dropdown').html() === '') {
        $('#navbar-dropdown').hide();
        $('#login-button').show();
    }

    // block UI until loaded
    window.showLoadingOverlay(true);


    // search function for label classes
    window.filterLabels = function(e) {
        // automatically select top hit on enter press
        var autoSelect = false;
        try {
            if(e.which === 13) {
                autoSelect = true;
            }
        } catch {}

        var searchBox = $('#labelclass-search-box');
        var keywords = searchBox.val();
        if(keywords != null && keywords != undefined) {
            keywords = keywords.split(/s[\s ]+/);
        }
        if(keywords.length === 0 || (keywords.length === 1 && keywords[0] === '')) keywords = null;
        window.labelClassHandler.filter(keywords, autoSelect);

        if(autoSelect) {
            // clear focus and search field after hitting enter
            searchBox.val('');
            searchBox.blur();
        }
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


    // project info
    var promise = $.ajax({
        url: 'getProjectInfo',
        method: 'get',
        success: function(data) {
            if(data.hasOwnProperty('info')) {
                // demo mode
                try {
                    window.demoMode = parseBoolean(data['info']['demoMode']);
                } catch {
                    window.demoMode = false;
                }
            }
        },
        error: function() {
            console.log('Error retrieving project info.');
            window.messager.addMessage('Error retrieving project info.', 'error', 0);
            window.demoMode = false;
        }
    });


    // authentication (+ login check)
    if(!window.demoMode) {
        promise = promise.then(function() {
            return $.ajax({
                url: 'getPermissions',
                method: 'post',
                success: function(data) {
                    try {
                        window.isAdmin = window.parseBoolean(data['permissions']['isAdmin']);
                    } catch {
                        window.location.href = '/login?redirect='+window.projectShortname+'/interface';
                    }
                },
                error: function() {
                    window.location.href = '/login?redirect='+window.projectShortname+'/interface';
                }
            });
        });
    }

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

            var height = Math.max(window.minImageWidth/aspectRatio, gallery.height() / numRows - numCols*24);   // subtract 24 pixels height for each image (footer)
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
            // var toolsWidth = $('#tools-container').width() + $('#viewport-controls').width();
            // $('#gallery').css({
            //     'width': 'calc(100vw - '+toolsWidth+'px)',
            // });
            // $('#classes-container').css('max-height', $('#tools-container').height() - $('#review-controls-container').height() - 76);

            // label class entries
            // $('#legend-entries').css('height', $('#classes-container').height() - 100); //gallery.height() - revCtrlHeight - 60 + 'px');   // -60 for search bar
            window.dataHandler.renderAll();

            // show or hide tooltip if scollable & adjust position
            $('#gallery-scroll-tooltip-bottom').css({
                'width': gallery.width(),
                'top': gallery.offset().top + gallery.outerHeight() - 25
            });
            window.galleryScrolled();
        }
        $(window).resize(windowResized);

        if(!window.demoMode) {
            $('#toolbox-divider').show();
            $('#review-controls-container').show();
            // // adjustable toolbox divider
            // $('#toolbox-divider').on({
            //     mousedown: function() {
            //         this.mouseDown = true;
            //     },
            //     mouseup: function() {
            //         this.mouseDown = false;
            //     },
            //     mousemove: function(event) {
            //         if(this.mouseDown) {
            //             // adjust space between legend and review toolboxes
            //             var container = $('#tools-container');
            //             var cHeight = container.height();
            //             var cTop = container.position().top + 75;   // 75 for padding at top
            //             var dividerPos = event.pageY;
            //             var divVal = 100*(dividerPos - cTop)/cHeight;
            //             $('#classes-container').css('height', divVal + '%');
            //             $('#review-controls-container').css('height', 100 - divVal + '%');
            //         }
            //     }
            // })

            // show or hide image review pane on title click
            $('#imorder-title').click(function() {
                var imorderBox = $('#imorder-box');
                if(imorderBox.is(':visible')) {
                    imorderBox.slideUp();
                } else {
                    imorderBox.slideDown();
                }
            });          
        }


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

        // login verification screen
        window.showVerificationOverlay = function(callback) {
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

            var overlayHtml = $('<h2>Renew Session</h2><div class="row fieldRow"><label for="password" class="col-sm">Password:</label><input type="password" name="password" id="password" required class="col-sm" /></div><div class="row fieldRow"><div class="col-sm"><div id="invalid-password" style="display:none;color:red;">invalid password entered</div><button id="abort" class="btn btn-sm btn-danger">Cancel</button><button id="confirm-password" class="btn btn-sm btn-primary float-right">OK</button></div></div>');
            window.showOverlay(overlayHtml, false, false);

            $('#abort').click(function() {
                window.location.href = '/login';
            })

            $('#confirm-password').click(function() {
                loginFun(callback);
            });
        }

        window.verifyLogin = function(callback) {
            return $.ajax({
                url: 'loginCheck',
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
        if(window.demoMode) {
            $('.dropdown-menu').hide();
            // $('.dropdown-toggle').hide();
        } else {
            window.onbeforeunload = function() {
                window.dataHandler._submitAnnotations(true);
            };
    
            $('#logout').click(function() {
                window.dataHandler._submitAnnotations(true);
                window.location.href = '/logout';
            });
        }
    });


    // AI backend
    promise = promise.then(function() {
        if(window.aiModelAvailable && window.aiControllerURI != null && ! window.demoMode) {
            // window.aiWorkerHandler = new AIWorkerHandler($('.ai-worker-entries'));
            //TODO: new workflow monitor
            window.wfMonitor = new WorkflowMonitor($('#ai-task-monitor'), window.isAdmin, 1000, 10000, 10000);    //TODO: query intervals
            $('#ai-worker-minipanel').click(function() {
                $('#ai-worker-panel').slideToggle();
            });
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
        if(!(window.getCookie('skipTutorial')) && !(window.annotationType === 'segmentationMasks'))     //TODO: implement tutorial for segmentation
            window.showTutorial(true);
    });
});