/*
    Sets up the frontend and loads all required parameters in correct order.

    2019-21 Benjamin Kellenberger
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

                        // show project admin shortcut buttons if admin
                        if(window.isAdmin) {
                            $('#project-config-links').show();
                        }

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
            'mouseenter',
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

            var height = Math.max(window.minImageWidth/aspectRatio, gallery.height() / numRows - numCols*48);   // subtract 48 pixels height for each image (footer)
            var width = Math.max(window.minImageWidth, gallery.width() / numCols);
            if(height > width/aspectRatio) {
                height = width/aspectRatio;
            } else {
                width = height*aspectRatio;
            }

            var style = {
                'min-width':  window.minImageWidth+'px',
                'width': width + 'px',
                //'height': height + 'px'
            };
            $('.entry').css(style);

            // update actual canvas width and height
            $('canvas').each(function() {
                this.width = width;
                this.height = height;
            });

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


        // toolboxes
        $('#toolbox-tab-container').children().on('click', function() {
            $('.toolbox-tab').removeClass('btn-primary');
            $('.toolbox-tab').addClass('btn-secondary');
            $(this).removeClass('btn-secondary');   //TODO: ugly
            $(this).addClass('btn-primary');
            $('.toolbox').hide();
            let tbId = $(this).attr('id').replace('tab-', '');
            $('#'+tbId).show();
        });
        if(window.demoMode) {
            $('#toolbox-tab-order').hide();
        }

        // overlay for meta key + pan to zoom
        let metaKey = 'Meta';
        if(navigator.userAgent.indexOf('Mac') != -1) {
            // macOS
            metaKey = '&#8984;';
        } else if(navigator.userAgent.indexOf('Win') != -1) {
            // Windows
            metaKey = '\u2756 (Windows key)';
        } else if(navigator.userAgent.indexOf("Linux") != -1) {
            // Linux
            metaKey = '&#8984; (Super key)';
        }
        $('#meta-key-zoom').html('Hold ' + metaKey + ' and scroll to zoom')

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

    // task monitor
    promise = promise.then(function() {
        window.taskMonitor = new TaskMonitor($('#task-panel'));
    });

    // AI backend
    promise = promise.then(function() {
        if(window.aiControllerURI != null && ! window.demoMode && (window.aiModelAvailable || window.isAdmin)) {
            window.wfMonitor = new WorkflowMonitor($('#ai-task-monitor'), $('#ai-minipanel-status'), window.isAdmin, window.aiModelAvailable, 1000, 10000, 10000);    //TODO: query intervals
            $('#ai-worker-minipanel').click(function() {
                $('#ai-worker-panel').slideToggle();
            });
            $('#ai-worker-minipanel').show();
        }
    });


    // load image batch
    promise = promise.then(function() {
        return window.dataHandler._loadFirstBatch();    //_loadNextBatch();
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