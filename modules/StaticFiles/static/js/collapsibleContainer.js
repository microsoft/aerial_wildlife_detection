/**
 * Markup with a clickable arrow that opens up a container, attached to it even
 * under move.
 * 
 * 2021 Benjamin Kellenberger
 */

class CollapsibleContainer {

    constructor(containerContents, placement) {
        
        // setup clickable markups
        this.anchorBox = $('<div class="collapsible-anchor"></div>');       // clickable anchor

        this.container = $('<div class="collapsible-container"></div>');    // container that expands and moves with anchor
        this.container.append($(containerContents));

        let self = this;
        this.anchorBox.on('click', function() {
            self.container.show();
        });
    }
}