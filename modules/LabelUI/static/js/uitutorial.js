/*
    Shows tooltips over interface controls at the start,
    unless already seen by the user.

    2019 Benjamin Kellenberger
*/

window.showTutorial = function(autostart) {
    window.setUIblocked(true);

    // list of identifiers andtext descriptions to be shown
    if(window.annotationType === 'labels') {
        var addAnnotationString = 'Then, click to assign the label. Click again or option-click to remove it.';
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
        [ '#gallery', 'View the next image(s) here.', 'left' ],
        [ '#legend-entries', 'Select the correct label class (or press its number on the keyboard).', 'bottom' ],
        [ '#add-annotation', 'Click to add a new annotation (hint: you can also use the W key).', 'top' ],
        [ '#gallery', addAnnotationString, 'left' ],
        [ '#gallery', changeAnnotationString, 'left' ],
        [ '#labelAll-button', 'Label everything with the foreground class (or press the A key)', 'top'],
        [ '#unsure-button', unsureString, 'top' ],
        [ '#remove-annotation', removeAnnotationString, 'top' ],
        [ '#clearAll-button', 'Remove all annotations at once (or press C)', 'top'],
        [ '#gallery', 'Temporarily hide predictions (annotations) by holding down the shift (control) key.', 'left' ],
        [ '#next-button', 'Satisfied with your annotations? Click "Next" (or press the right arrow key).', 'top' ],
        [ '#previous-button', 'Want to review the last image(s)? Click "Previous" (or press the left arrow key).', 'top' ],
        // [ '#ai-worker-panel', 'View the tasks and progress of the AI worker(s) by clicking here.' ]  //TODO: doesn't work; perhaps z-index problem?
    ];

    var index = -2;
    var nextTooltip = function() {
        window.setUIblocked(true);

        if(index >= 0) {
            // if(interfaceElements[index][0] === '#tools-container') {
            //     // hide class drawer
            //     let offset = -$('#tools-container').outerWidth() + 40;
            //     $('#tools-container').animate({
            //         right: offset
            //     });

            //     // re-enable original mouseleave command (TODO: ugly)
            //     $('#tools-container').on('mouseleave', function() {
            //         if(window.uiBlocked) return;
            //         let offset = -$(this).outerWidth() + 40;
            //         $('#tools-container').animate({
            //             right: offset
            //         });
            //     });
            // } else 
            if(interfaceElements[index][0] === '#ai-worker-panel') {
                // minimize AI panel
                $('#ai-worker-panel').slideUp();
            }
            $(interfaceElements[index][0]).tooltip('dispose');
        }

        do {
            index += 1;
            if(index >= interfaceElements.length) {
                // done with tooltips
                $(window).off('click', advance);
                window.setUIblocked(false);
                window.setCookie('skipTutorial', true, 365);
                return;
            }
        } while($(interfaceElements[index][0]).length == 0);
        
        
        $([document.documentElement, document.body]).animate({
            scrollTop: $(interfaceElements[index][0]).offset().top
        }, 1000);

        // if(interfaceElements[index][0] === '#tools-container') {
        //     // show class drawer
        //     $('#tools-container').animate({
        //         right: 0
        //     }, 500, function() {
        //         $(interfaceElements[index][0]).tooltip({
        //             title: interfaceElements[index][1],
        //             placement: interfaceElements[index][2]
        //         }).off("mouseover mouseout mouseleave").tooltip('show');
        //     });

        // } else 
        if(interfaceElements[index][0] === '#ai-worker-panel') {
            // show AI panel
            $('#ai-worker-panel').slideDown();

        } else {
            $(interfaceElements[index][0]).tooltip({
                title: interfaceElements[index][1],
                placement: interfaceElements[index][2]
            }).off("mouseover mouseout mouseleave").tooltip('show');
        }
    }


    var advance = function() {
        if(index === -2) {
            var welcomeContents = $('<div style="overflow-y:auto"></div>');
            welcomeContents.load('static/templates/tutorial_welcome.html');
            window.showOverlay(welcomeContents, true);
            index += 1;
        } else {
            window.showOverlay(null, false, true);
            nextTooltip();
        }
    }

    $(window).on('click', advance);
    index = -2;

    if(autostart) {
        advance();
    }
}