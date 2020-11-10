/**
 * Displays a row of initial letters that can be picked
 * and sends out events to connected listeners. Used for
 * e.g. users tables.
 * 
 * 2020 Benjamin Kellenberger
 */

class LetterPicker {

    constructor(domElement, data) {
        this.domElement = $(domElement);
        this.data = data;

        this.letterArray = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0'.split('');

        this.selectedLetters = {};

        // optional settings
        if(this.data.hasOwnProperty('allowMultipleSelection')) {
            this.allowMultipleSelection = this.data['allowMultipleSelection'];
        }

        // event handlers
        this.events = {
            'click': [],
            'select': [],
            'deselect': []
        }

        // set data
        this.setData(this.data['data']);
    }

    _setup_markup() {
        this.domElement.empty();
        let baseElement = $('<div class="letterpicker"></div>');
        let self = this;
        for(var l=0; l<this.letterArray.length; l++) {
            let letter = (this.letterArray[l] === '0' ? '#' : this.letterArray[l]);
            let letterMarkup = $('<div id="letterpicker-letter-'+this.letterArray[l]+'" class="letterpicker-letter">'+letter+'</div>');
            if(this.letterMap[this.letterArray[l]]) {
                letterMarkup.addClass('letterpicker-letter-available');
            } else {
                letterMarkup.addClass('letterpicker-letter-disabled');
            }
            letterMarkup.on('click', function() {
                self._letterClicked($(this).attr('id').replace('letterpicker-letter-', ''));
            });
            baseElement.append(letterMarkup);
        }
        this.domElement.append(baseElement);
    }

    _create_letter_map() {
        this.letterMap = this.letterArray.reduce(function(acc, cur, i) {
            acc[cur] = false;
            return acc;
        }, {});
    }

    _letterClicked(letter) {
        let selected = true;
        if(this.selectedLetters.hasOwnProperty(letter)) {
            // deselect
            $('#letterpicker-letter-' + letter).removeClass('letterpicker-letter-selected');
            delete this.selectedLetters[letter];
            selected = false;
        } else {
            // select
            if(!this.allowMultipleSelection) {
                // deselect others first
                for(var key in this.selectedLetters) {
                    console.log('deselecting ' + key)
                    $('#letterpicker-letter-' + key).removeClass('letterpicker-letter-selected');
                    delete this.selectedLetters[key];
                }
            }
            this.selectedLetters[letter] = true;
            $('#letterpicker-letter-' + letter).addClass('letterpicker-letter-selected');
        }
        for(var c=0; c<this.events['click'].length; c++) {
            this.events['click'][c]({letter: letter, selected: selected});
        }
    }

    on(event, callback) {
        if(this.events.hasOwnProperty(event)) {
            this.events[event].push(callback);
        }
    }

    setData(data) {
        this._create_letter_map();
        
        // determine which letters are to be found in data
        if(data !== undefined && Array.isArray(data)) {
            for(var d=0; d<data.length; d++) {
                let firstLetter = data[d].slice(0,1).toUpperCase();
                if(!this.letterMap.hasOwnProperty(firstLetter)) {
                    this.letterMap['#'] = true;     // non-alphabetical characters
                } else {
                    this.letterMap[firstLetter] = true;
                }
            }
        }
        
        // now create markup
        this._setup_markup();
    }

    getSelected() {
        return Object.keys(this.selectedLetters);
    }
}