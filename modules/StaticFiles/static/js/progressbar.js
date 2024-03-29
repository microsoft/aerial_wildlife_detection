/*
 *  2020-21 Benjamin Kellenberger
 */
class ProgressBar {
    constructor(visible, value, max, indefinite) {
        this.value = value;
        this.max = max;
        this.indefinite = indefinite;

        this.markup = $('<div class="progressbar"></div>');
        this.pbarInd = $('<div class="progressbar-filler progressbar-active"></div>');
        this.markup.append(this.pbarInd);
        this.set(visible, this.value, this.max, this.indefinite);
    }

    set(visible, value, max, indefinite) {
        if(visible !== undefined) {
            if(!visible) {
                let displayProp = this.markup.css('display');
                if(displayProp.trim().toLowerCase() !== 'none') {
                    this.displayProp = this.markup.css('display');
                } else {
                    this.displayProp = undefined;
                }
                this.markup.css('display', 'none');
            } else {
                if(typeof(this.displayProp) === 'string') {
                    this.markup.css('display', this.displayProp);
                } else {
                    this.markup.show();
                }
            }
        }
        if(max !== undefined) {
            this.max = max;
        }
        if(value !== undefined) {
            this.value = value;
        }
        if(indefinite !== undefined) {
            this.indefinite = indefinite;
        }
        if(this.indefinite) {
            var newWidthPerc = 100;
            this.pbarInd.css('width', newWidthPerc + '%');
        } else {
            var newWidthPerc = (100*(this.value / this.max));
            this.pbarInd.animate({
                'width': newWidthPerc + '%'
            }, 1000);
        }
    }

    getValue() {
        return this.value;
    }

    getMax() {
        return this.max;
    }

    getMarkup() {
        return this.markup;
    }
}