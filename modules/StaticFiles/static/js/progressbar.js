/*
 *  2020 Benjamin Kellenberger
 */

class ProgressBar {
    constructor(visible, value, max) {
        this.value = value;
        this.max = max;

        this.markup = $('<div class="progressbar"></div>');
        this.pbarInd = $('<div class="progressbar-filler progressbar-active"></div>');
        this.markup.append(this.pbarInd);
        if(visible !== undefined) {
            this.markup.css('visibility', (visible? 'visible' : 'hidden'));
        }
    }

    set(visible, value, max) {
        if(visible !== undefined) {
            this.markup.css('visibility', (visible? 'visible' : 'hidden'));
        }
        if(max !== undefined) {
            this.max = max;
        }
        if(value !== undefined) {
            this.value = value;
        }
        var newWidthPerc = (100*(this.value / this.max));
        this.pbarInd.animate({
            'width': newWidthPerc + '%'
        }, 1000);
    }

    getMarkup() {
        return this.markup;
    }
}