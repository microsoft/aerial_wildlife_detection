/*
    Shows tooltips over interface controls at the start,
    unless already seen by the user.

    2019 Benjamin Kellenberger
*/

window.showTutorial = function() {

    // list of identifiers andtext descriptions to be shown
    let interfaceElements = [
        [ '#gallery', 'View the next image(s) here.' ],
        [ '#legend-entries', 'Select the correct label class.' ],
        [ '#add-annotation', 'Click to add a new annotation (hint: you can also use the W key).' ],
        [ '#gallery', 'Click to assign label. Click again or option-click to assign "none".' ],       //TODO
        [ '#remove-annotation', 'Click (or press R), then click into the annotation to remove it.' ],

        [ '#next-button', 'Satisfied with your annotations? Click "Next" (or press the right arrow key).' ],
        [ '#previous-button', 'Want to review the last image(s)? Click "Previous" (or press the left arrow key).' ]
    ];

    var index = 0;
    var clickEvent = null;
    var nextTooltip = function() {
        $(interfaceElements[index][0]).tooltip('dispose');
        do {
            index += 1;
            if(index >= interfaceElements.length) {
                // done with tooltips
                $(window).off(clickEvent);
                return;
            }
        } while($(interfaceElements[index][0]).length == 0);

        $(interfaceElements[index][0]).tooltip({
            title: interfaceElements[index][1]
        }).off("mouseover mouseout mouseleave").tooltip('show');
    }

    clickEvent = $(window).click(function() {
        nextTooltip();
    })

    nextTooltip();
}