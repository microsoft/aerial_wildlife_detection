/*
    Shows tooltips over interface controls at the start,
    unless already seen by the user.

    2019 Benjamin Kellenberger
*/

window.showTutorial = function(autostart) {
    window.setUIblocked(true);

    // list of identifiers andtext descriptions to be shown
    if(window.annotationType === 'labels') {
        var addAnnotationString = 'Then, click to assign label. Click again or option-click to remove it.';
        var changeAnnotationString = 'To change the class, select the correct class first and then click into the image.';
        var unsureString = 'Not sure? Click (or press U) and click the difficult image (tip: you can also hover over the difficult image and press U directly).';
        var removeAnnotationString = 'Click (or press R), then click into the image to remove its label.';
    } else if(window.annotationType === 'points') {
        var addAnnotationString = 'Then, click into the image to put a point at the given position.';
        var changeAnnotationString = 'To change the class of a point, select the correct class first and then click the point.';
        var unsureString = 'Not sure? Select all difficult annotations and click here (or press the U key)';
        var removeAnnotationString = 'Click (or press R), then click into the annotation to remove it. Hint: remove selected annotations directly with the Del key.';
    } else if(window.annotationType === 'boundingBoxes') {
        var addAnnotationString = 'Then, click and drag to draw a bounding box in the image.';
        var changeAnnotationString = 'To change the class of a bounding box, select the correct class first and then click the bounding box.';
        var unsureString = 'Not sure? Select all difficult annotations and click here (or press the U key)';
        var removeAnnotationString = 'Click (or press R), then click into the annotation to remove it. Hint: remove selected annotations directly with the Del key.';
    }
    let interfaceElements = [
        [ '#gallery', 'View the next image(s) here.' ],
        [ '#tools-container', 'Select the correct label class (or press its number on the keyboard).' ],
        [ '#add-annotation', 'Click to add a new annotation (hint: you can also use the W key).' ],
        [ '#gallery', addAnnotationString ],
        [ '#gallery', changeAnnotationString ],
        [ '#labelAll-button', 'Label everything with the foreground class (or press the A key)'],
        [ '#unsure-button', unsureString ],
        [ '#remove-annotation', removeAnnotationString ],
        [ '#clearAll-button', 'Remove all annotations at once (or press C)'],
        [ '#next-button', 'Satisfied with your annotations? Click "Next" (or press the right arrow key).' ],
        [ '#previous-button', 'Want to review the last image(s)? Click "Previous" (or press the left arrow key).' ],
        [ '#ai-worker-panel', 'View the tasks and progress of the AI worker(s) by clicking here.' ]
    ];

    var index = -1;
    var nextTooltip = function() {
        window.setUIblocked(true);

        if(index >= 0) {
            if(interfaceElements[index][0] === '#tools-container') {
                // hide class drawer
                let offset = -$('#tools-container').outerWidth() + 40;
                $('#tools-container').animate({
                    right: offset
                });

                // re-enable original mouseleave command (TODO: ugly)
                $('#tools-container').on('mouseleave', function() {
                    if(window.uiBlocked) return;
                    let offset = -$(this).outerWidth() + 40;
                    $('#tools-container').animate({
                        right: offset
                    });
                });
            } else if(interfaceElements[index][0] === '#ai-worker-panel') {
                // minimize AI panel
                $('#ai-tasks-entries').slideUp();
                $('#ai-tasks-header').toggleClass('expanded');
            }
            $(interfaceElements[index][0]).tooltip('dispose');
        }

        do {
            index += 1;
            if(index >= interfaceElements.length) {
                // done with tooltips
                $(window).off('click', nextTooltip);
                window.setUIblocked(false);
                window.setCookie('skipTutorial', true, 365);
                return;
            }
        } while($(interfaceElements[index][0]).length == 0);
        
        
        $([document.documentElement, document.body]).animate({
            scrollTop: $(interfaceElements[index][0]).offset().top
        }, 1000);

        if(interfaceElements[index][0] === '#tools-container') {
            // show class drawer
            $('#tools-container').animate({
                right: 0
            }, 500, function() {
                $(interfaceElements[index][0]).tooltip({
                    title: interfaceElements[index][1]
                }).off("mouseover mouseout mouseleave").tooltip('show');
            });

        } else if(interfaceElements[index][0] === '#ai-worker-panel') {
            // show AI panel
            $('#ai-tasks-entries').slideDown();
            $('#ai-tasks-header').toggleClass('expanded');

        } else {
            $(interfaceElements[index][0]).tooltip({
                title: interfaceElements[index][1]
            }).off("mouseover mouseout mouseleave").tooltip('show');
        }
    }


    if(autostart) {
        // show welcome instructions
        var welcomeContents = '<h2>Welcome!</h2>';      //TODO
        welcomeContents += '<div style="border:' + window.styles.predictions.lineWidth + 'px solid blue; width:100px;height:60px;"></div>';
        window.showOverlay(welcomeContents);

        // show tooltips
        $(window).on('click', nextTooltip);
        nextTooltip();
    } else {
        //TODO
        $(window).on('click', nextTooltip);
    }
}