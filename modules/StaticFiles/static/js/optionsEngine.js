/*
    Engine that parses, renders, and returns JSON-encoded options,
    e.g. for AI models.

    2020 Benjamin Kellenberger
 */

var RESERVED_KEYWORDS = [
    'id', 'name', 'description', 'type', 'min', 'max', 'value', 'style', 'options'
];


function parseOption(optionID, optionSettings, additionalOptions) {
    if(typeof(optionID) !== 'string') {
        if(optionSettings.hasOwnProperty('id')) {
            optionID = optionSettings['id'];
        } else {
            optionID = Date.now().toString();
        }
    }
    let name = (typeof(optionSettings['name']) === 'string' &&
                optionSettings['name'].length > 0 ? optionSettings['name'] : optionID);
    let description = optionSettings['description'];
    let type = optionSettings['type'];
    if(optionSettings.hasOwnProperty('style')) {
        if(typeof(additionalOptions) === 'object') {
            additionalOptions = {...additionalOptions, ...optionSettings['style']};
        } else {
            additionalOptions = optionSettings['style'];
        }
    }
    if(typeof(additionalOptions) !== 'object') {
        additionalOptions = {};
    }

    // get number of subfields
    var numSubfields = 0;
    for(var key in optionSettings) {
        if(!RESERVED_KEYWORDS.includes(key)) {
            numSubfields++;
        }
    }

    // find out what kind of option it is
    if(type === 'select') {
        // drop-down selection
        return new SelectOption(optionID, name, description, optionSettings['options'], optionSettings['value'], additionalOptions);

    } else if(type === 'list') {
        // list option
        return new ListOption(optionID, name, description, optionSettings['options'], optionSettings['value'], additionalOptions);

    } else if(optionSettings.hasOwnProperty('value')) {
        // primitives
        additionalOptions = {...additionalOptions, ...{'showBorder': false}};
        let value = optionSettings['value'];
        if(typeof(value) === 'boolean') {
            return new CheckboxOption(optionID, name, description, value, additionalOptions);
        } else if(typeof(value) === 'number') {
            return new NumberOption(optionID, name, description, value,
                optionSettings['min'], optionSettings['max'], additionalOptions);
        } else {
            return new SimpleOption(optionID, name, description, value, additionalOptions);
        }

    } else if(optionSettings.hasOwnProperty('min') || optionSettings.hasOwnProperty('max')) {
        // number type
        additionalOptions = {...additionalOptions, ...{'showBorder': false}};
        return new NumberOption(optionID, name, description, optionSettings['value'],
            optionSettings['min'], optionSettings['max'], additionalOptions);

    } else if(numSubfields>0 && typeof(optionSettings) === 'object') {
        // nested options
        return new ComposedOption(optionID, name, description, optionSettings, additionalOptions);

    } else {
        return new SimpleOption(optionID, name, description, name, additionalOptions);
    }
}



class AbstractOption {

    constructor(id, name, description, additionalOptions) {
        this.id = id;
        this.name = name;
        this.description = description;
        this.additionalOptions = additionalOptions;
        if(typeof(additionalOptions) !== 'object') {
            this.additionalOptions = {};
        }
        if(typeof(this.name) !== 'string' || this.name.length === 0) {
            this.name = this.id;
        }
    }

    setupMarkup() {
        this.markup = $('<div class="option-container"></div>');
        if(this.additionalOptions['showBorder']) {
            this.markup.addClass('option-container-border');
        }
    }

    _addName(makeHeader) {
        if(this.additionalOptions['hideHeader']) return;
        if(makeHeader) {
            var header = $('<h3 class="option-header">'+this.name+'</h3>');
        } else {
            var header = $('<div class="option-header">'+this.name+'</div>');
        }
        if(this.additionalOptions['selectable']) {
            header.css('cursor', 'pointer');
            var self = this;
            header.on('click', function() {
                self.setSelected(!self.getSelected());
            });
        }
        this.markup.append(header);
    }

    _addDescription() {
        if(this.additionalOptions['hideDescription']) return;
        if(typeof(this.description) === 'string' && this.description.length>0) {
            this.markup.append($('<div class="option-description">'+this.description+'</div>'));
        }
    }

    getSelected() {
        if(!this.additionalOptions['selectable']) return false;
        return this.markup.hasClass('option-selected');
    }

    setSelected(selected) {
        if(!this.additionalOptions['selectable']) return;
        if(selected) {
            this.markup.addClass('option-selected');
        } else {
            this.markup.removeClass('option-selected');
        }
    }

    getOptions(minimal) {
        let options = {
            id: this.id
        };
        if(!minimal) {
            options['name'] = this.name;
            if(typeof(this.description) === 'string') {
                options['description'] = this.description;
            }
            if(Object.keys(this.additionalOptions).length > 0) {
                options['style'] = this.additionalOptions;
            }
        }
        return options;
    }
}



class SimpleOption extends AbstractOption {

    constructor(id, name, description, value, additionalOptions) {
        super(id, name, description, additionalOptions);
        if(typeof(value) === 'string') {
            // get from global options if there is a match
            if(window.globals.hasOwnProperty(value)) {
                value = JSON.parse(JSON.stringify(window.globals[value]));
            }
        }
        this.value = value;
        this.setupMarkup();
    }

    setupMarkup() {
        super.setupMarkup();
        this._addName();
        this._addDescription();
    }
}



class CheckboxOption extends AbstractOption {

    constructor(id, name, description, value, additionalOptions) {
        super(id, name, description, additionalOptions);
        if(typeof(value) === 'string') {
            // identifier; get from global definitions
            if(window.globals.hasOwnProperty(value)) {
                value = JSON.parse(JSON.stringify(window.globals[value]));
            }
        }
        this.value = value;
        this.setupMarkup();
    }

    setupMarkup() {
        super.setupMarkup();
        this.checkbox = $('<input type="checkbox" id="'+this.id+'_chck" />');
        this.checkbox.prop('checked', this.value);
        this.markup.append(this.checkbox);
        this.markup.append($('<label for="'+this.id+'_chck">'+this.name+'</label>'));
        this._addDescription();
    }

    getOptions(minimal) {
        let options = super.getOptions(minimal);
        options['value'] = this.checkbox.prop('checked');
        return options;
    }
}



class NumberOption extends AbstractOption {

    constructor(id, name, description, value, min, max, numberType, additionalOptions) {
        super(id, name, description, additionalOptions);
        if(typeof(value) === 'string') {
            // identifier; get from global definitions
            if(window.globals.hasOwnProperty(value)) {
                value = JSON.parse(JSON.stringify(window.globals[value]));
            }
        }
        this.min = parseFloat(min);
        if(typeof(this.min) !== 'number') {
            this.min = 0;
        }
        this.max = parseFloat(max);
        if(typeof(this.max) !== 'number') {
            this.max = 1e9;
        }
        this.value = parseFloat(value);
        if(typeof(this.value) !== 'number' || isNaN(this.value)) {
            this.value = this.min;
        }
        this.numberType = numberType;
        if(!['int', 'float'].includes(numberType)) {
            // try to guess number type from given value
            if(this.value === +this.value && this.value !== (this.value|0)) {
                this.numberType = 'float';
            } else {
                this.numberType = 'int';
            }
        }
        this.setupMarkup();
    }

    setupMarkup() {
        super.setupMarkup();
        this.markup.append($('<span class="option-name">'+this.name+'</span>'));
        let step = (this.numberType === 'float' ? 'any' : '1');
        this.input = $('<input type="number" class="option-number-field" min="'+this.min+'" max="'+this.max+'" value="'+this.value+'" step="'+step+'" />');

        if(this.additionalOptions['slider']) {
            var self = this;
            this.slider = $('<input type="range" min="'+this.min+'" max="'+this.max+'" value="'+this.value+'" step="'+step+'" />');
            this.slider.on('input', function() {
                self.input.val($(this).val());
            });
            this.input.on('input', function() {
                self.slider.val($(this).val());
            });
            this.markup.append(this.slider);
        }

        this.markup.append(this.input);
        this._addDescription();
    }

    getOptions(minimal) {
        let options = super.getOptions(minimal);
        let min = parseFloat(this.input.prop('min'));
        let max = parseFloat(this.input.prop('max'));
        let value = Math.max(min, Math.min(max, parseFloat(this.input.val())));
        if(this.numberType === 'int') {
            min = parseInt(min);
            max = parseInt(max);
            value = parseInt(value);
        }
        if(!minimal) {
            options['min'] = min;
            options['max'] = max;
        }
        options['value'] = value;
        return options;
    }
}



class SelectOption extends AbstractOption {

    constructor(id, name, description, options, defaultOption, additionalOptions) {
        super(id, name, description, additionalOptions);
        this.options = [];
        this.optionsIndex = {};
        if(typeof(options) === 'string') {
            // identifier; get from global definitions
            if(window.globals.hasOwnProperty(options)) {
                options = JSON.parse(JSON.stringify(window.globals[options]));
            }
        }
        this.defaultOption = defaultOption;
        if(typeof(this.defaultOption) === 'object') {
            // default option provided with parameters; parse and extract ID
            // this.defaultOption = parseOption(this.defaultOption.id, this.defaultOption, undefined);
            var defaultOptionID = this.defaultOption.id;
        } else {
            var defaultOptionID = defaultOption;
        }
        var defaultFound = false;
        for(var o in options) {
            let optionID = o;
            if(typeof(options[o]) === 'object' && options[o].hasOwnProperty('id')) {
                optionID = options[o]['id'];
            }
            if(RESERVED_KEYWORDS.includes(optionID)) continue;
            if(optionID === defaultOptionID) {
                // default option
                defaultFound = true;
                if(typeof(this.defaultOption) !== 'string') {
                    // default option provided with arguments; parse
                    var parsedOption = parseOption(optionID, this.defaultOption, {hideHeader:true});
                } else {
                    var parsedOption = parseOption(optionID, options[o], {hideHeader:true});
                }
            } else {
                var parsedOption = parseOption(optionID, options[o], {hideHeader:true});
            }
            this.options.push(parsedOption);
            this.optionsIndex[optionID] = this.options.length-1;
        }
        if(!defaultFound && typeof(defaultOptionID) === 'string') {
            console.warn('WARNING: default option ID "'+defaultOptionID+'" not found; ignoring...');
            this.defaultOption = undefined;
        }
        this.setupMarkup();
    }

    setupMarkup() {
        super.setupMarkup();
        var self = this;
        this.select = $('<select class="options-select"></select>');
        for(var o=0; o<this.options.length; o++) {
            var option = $('<option value="'+this.options[o].id+'">'+this.options[o].name+'</option>');
            this.select.append(option);
        }
        if(typeof(this.defaultOption) === 'object') {
            // default option provided with parameters; extract ID
            this.select.val(this.defaultOption.id);
        } else if(typeof(this.defaultOption) === 'string') {
            // string provided
            this.select.val(this.defaultOption);
        }
        this.selOptionDiv = $('<div style="margin-left:20px;display:none"></div>');
        this.select.on('change', function() {
            let selectedID = $(this).val();
            let selected = self.options[self.optionsIndex[selectedID]];
            self.selOptionDiv.empty();
            if(selected.markup !== undefined && selected.markup.children().length>0) {
                self.selOptionDiv.append(selected.markup);
                self.selOptionDiv.show();
            } else {
                self.selOptionDiv.hide();
            }
        });
        this.select.trigger('change');
        if(this.additionalOptions['inline']) {
            let container = $('<div class="options-inline"></div>');
            container.append($('<span class="option-name">'+this.name+'</span>'));
            container.append(this.select);
            this.markup.append(container);
            this._addDescription();
            this.markup.append(this.selOptionDiv);
        } else {
            this._addName(true);
            this._addDescription();
            this.markup.append(this.select);
            this.markup.append(this.selOptionDiv);
        }
    }

    getOptions(minimal) {
        let options = super.getOptions(minimal);
        options['type'] = 'select';
        if(!minimal) {
            let availables = [];
            for(var o=0; o<this.options.length; o++) {
                availables.push(this.options[o].getOptions(minimal));
            }
            options['options'] = availables;
        }
        options['value'] = this.options[this.optionsIndex[this.select.val()]].getOptions(minimal);
        return options;
    }
}



class ListOption extends AbstractOption {

    constructor(id, name, description, options, defaults, additionalOptions) {
        super(id, name, description, additionalOptions);
        if(typeof(options) === 'string') {
            // identifier; get from global definitions
            if(window.globals.hasOwnProperty(options)) {
                options = JSON.parse(JSON.stringify(window.globals[options]));
            }
        }
        this.fixedList = (options === undefined || options === null ||
                Object.keys(options).length === 0);

        this.options = {};
        if(!this.fixedList) {
            // parse available options only if it's not a fixed list
            if(Array.isArray(options)) {
                for(var o=0; o<options.length; o++) {
                    var optionID = o;
                    if(typeof(options[o]) === 'string') {
                        // possible identifier; get from global definitions
                        optionID = options[o];
                        if(window.globals.hasOwnProperty(options[o])) {
                            options[o] = JSON.parse(JSON.stringify(window.globals[options[o]]));
                        }
                    } else if(options[o].hasOwnProperty('id')) {
                        optionID = options[o]['id'];
                    }
                    if(RESERVED_KEYWORDS.includes(optionID)) {
                        continue;
                    }
                    // let option = parseOption(optionID, options[o], {'showBorder': false, 'selectable': true});
                    this.options[optionID] = options[o];
                }
            } else {
                for(var key in options) {
                    if(RESERVED_KEYWORDS.includes(key)) {
                        continue;
                    }
                    if(typeof(options[key]) === 'string') {
                        // possible identifier; get from global definitions
                        if(window.globals.hasOwnProperty(options[key])) {
                            options[key] = JSON.parse(JSON.stringify(window.globals[options[key]]));
                        }
                    }
                    // let option = parseOption(key, options[key], {'showBorder': false, 'selectable': true});
                    this.options[key] = options[key];
                }
            }
        }
        
        // parse defaults
        this.selectedOptions = [];
        if(typeof(defaults) === 'string' && window.globals.hasOwnProperty(defaults)) {
            // identifier; get from global definitions
            defaults = JSON.parse(JSON.stringify(window.globals[defaults]));
        }
        if(defaults === undefined || defaults === null) {
            defaults = [];
        }
        for(var d=0; d<defaults.length; d++) {
            var optionID = undefined;
            if(typeof(defaults[d]) === 'string') { 
                optionID = defaults[d];
                if(window.globals.hasOwnProperty(defaults[d])) {
                    // replace with global option
                    defaults[d] = JSON.parse(JSON.stringify(window.globals[defaults[d]]));
                }
            }
            
            let option = parseOption(d, defaults[d], {'showBorder': false, 'selectable': true});
            if(optionID === undefined) {
                optionID = option.id;
            }
            if(this.options.hasOwnProperty(optionID) || this.fixedList) {
                // default is part of options; or else we have a fixed list
                this.selectedOptions.push(option);
            } else {
                console.warn('WARNING: default entry "'+optionID+'" for option "'+this.id+'" is not a valid option; skipping...');
            }
        }
        this.setupMarkup();
    }

    setupMarkup() {
        super.setupMarkup();
        this._addName(true);
        this._addDescription();
        this.list = $('<div class="options-list"></div>');
        for(var s=0; s<this.selectedOptions.length; s++) {
            let itemMarkup = this.selectedOptions[s].markup;
            itemMarkup.addClass('options-list-entry');
            this.list.append(itemMarkup);
        }
        this.markup.append(this.list);

        // selection, add and remove buttons
        if(!this.fixedList) {
            var self = this;
            this.addOptionSelect = $('<select style="margin-bottom:5px"></select>');
            for(var key in this.options) {
                var name = this.options[key]['name'];
                if(typeof(name) !== 'string' || name.length === 0) {
                    name = key;
                }
                this.addOptionSelect.append($('<option value="'+key+'">'+name+'</option>'));
            }
            this.markup.append('<div>Add item:</div>');
            this.markup.append(this.addOptionSelect);
            let addOptionButton = $('<button class="btn btn-sm btn-primary">+</button>');
            addOptionButton.on('click', function() {
                let selectedOptionType = $(self.addOptionSelect).val();
                let selectedOption = parseOption(selectedOptionType, self.options[selectedOptionType], {'showBorder': false, 'selectable': true});
                self.selectedOptions.push(selectedOption);
                self.list.append(selectedOption.markup);
            });
            this.markup.append(addOptionButton);
            let removeOptionButton = $('<button class="btn btn-sm btn-warning">-</button>');
            removeOptionButton.on('click', function() {
                // remove selected
                for(var item in self.selectedOptions) {
                    if(self.selectedOptions[item].getSelected()) {
                        self.selectedOptions[item].markup.remove();
                        delete self.selectedOptions[item];
                    }
                }
            });
            this.markup.append(removeOptionButton);
        }
    }

    getOptions(minimal) {
        let options = super.getOptions(minimal);
        options['type'] = 'list';
        if(!this.fixedList) {
            options['options'] = {};
            for(var key in this.options) {
                options['options'][key] = this.options[key]
            }
        }
        options['value'] = [];
        for(var i=0; i<this.selectedOptions.length; i++) {
            options['value'].push(this.selectedOptions[i].getOptions(minimal));
        }
        return options;
    }
}



class ComposedOption extends AbstractOption {

    constructor(id, name, description, fields, additionalOptions) {
        super(id, name, description, additionalOptions);

        // parse fields
        this.fields = [];
        for(var f in fields) {
            if(!RESERVED_KEYWORDS.includes(f)) {
                let field = parseOption(f, fields[f], {'showBorder': true});
                this.fields.push(field);
            }
        }
        this.setupMarkup();
    }

    setupMarkup() {
        super.setupMarkup();
        this.markup.addClass('option-container-composed');
        this._addName(true);
        this._addDescription();
        let fieldsDiv = $('<div></div>');
        if(this.additionalOptions['inline']) {
            fieldsDiv.addClass('options-inline')
        }
        for(var f=0; f<this.fields.length; f++) {
            fieldsDiv.append(this.fields[f].markup);
        }
        this.markup.append(fieldsDiv);
    }

    getOptions(minimal) {
        let options = super.getOptions(minimal);
        for(var i=0; i<this.fields.length; i++) {
            options[this.fields[i].id] = this.fields[i].getOptions(minimal);
        }
        return options;
    }
}


function _flatten_globals(options, defs) {
    if(options === undefined) return;
    if(defs === undefined) {
        defs = options;
    }
    if(typeof(options) !== 'object') return
    for(var key in options) {
        if(RESERVED_KEYWORDS.includes(key)) {
            // reserved keywords
            continue

        } else if(typeof(key) === 'string') {
            defs[key] = JSON.parse(JSON.stringify(options[key]));
            if(!Array.isArray(options)) {
                defs[key]['id'] = key;
            }
            _flatten_globals(options[key], defs);
        }
    }
    return defs;
}



function _fill_globals(options, defs) {
    if(typeof(defs) !== 'object' || defs === undefined || defs === null) {
        return options;
    }
    if(['string', 'number', 'boolean'].includes(typeof(options))) {
        if(defs.hasOwnProperty(options)) {
            return JSON.parse(JSON.stringify(defs[options]));
        } else {
            // no match found; return string itself
            return options;
        }

    } else if(Array.isArray(options) || typeof(options) === 'object') {
        for(var key in options) {
            if(RESERVED_KEYWORDS.includes(key)) {
                // reserved keywords
                continue
            }
            // also check value
            if(['string', 'number', 'boolean'].includes(typeof(options[key]))) {
                if(defs.hasOwnProperty(options[key])) {
                    options[key] = JSON.parse(JSON.stringify(defs[options[key]]));
                }
            } else {
                options[key] = _fill_globals(options[key], defs);
            }
        }
        return options;
        
    } else {
        // unknown type
        console.warn('WARNING: unknown type for entry "'+options+'".');
        return options;
    }
}



class OptionsEngine {

    constructor(domElement, options) {
        this.domElement = $(domElement);

        this.setOptions(options);
    }

    setOptions(options) {
        // clear everything
        this.textarea = undefined;
        this.domElement.empty();
        this.defs = undefined;
        this.rootOption = undefined;

        if(options === null || options === undefined) return;

        // try to parse options
        try {
            if(typeof(options) !== 'object') {
                options = JSON.parse(options);
            }

            // expand and fill-in defaults from global definitions
            this.defs = options['defs'];
            if(this.defs === undefined || this.defs === null) {
                this.defs = {};
            }
            let defs = _flatten_globals(JSON.parse(JSON.stringify(this.defs)));
            defs = _fill_globals(defs, defs);
            window.globals = defs;
            let opts = JSON.parse(JSON.stringify(options['options']));
            opts = _fill_globals(opts, defs);

            // parse options
            let baseContainer = $('<div></div>');
            this.rootOption = new ComposedOption('options', 'Options', undefined, opts, {'hideHeader':true, 'showBorder': true});
            baseContainer.append(this.rootOption.markup);
            this.domElement.append(baseContainer);

        } catch(error) {
            console.info('Could not parse options (error: "'+error+'"), showing text area instead.');

            this.textarea = $('<textarea class="options-textarea"></textarea>');
            if(typeof(options) === 'object') {
                options = JSON.stringify(options, null, 2);
            }
            this.textarea.val(options);
            this.domElement.append(this.textarea);
        }
    }

    getOptions(minimal) {
        let options = {};

        if(this.textarea !== undefined) {
            // had to do a fallback; just return text area contents
            try {
                return JSON.parse(this.textarea.val());
            } catch {
                return this.textarea.val();
            }
        }

        if(!minimal) {
            options['defs'] = this.defs;
            if(this.rootOption !== undefined) {
                options['options'] = this.rootOption.getOptions(minimal);
            }
        } else if(this.rootOption !== undefined) {
            options = this.rootOption.getOptions(minimal);
        }
        
        return options;
    }
}