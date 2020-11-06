/**
 * Displays a folder hierarchy as a tree.
 * 
 * 2020 Benjamin Kellenberger
 */


class DirectoryElement {

    constructor(name, childDefinitions, parent, browser) {
        this.name = name;
        this.parent = parent;
        this.browser = browser;
        if(typeof(parent.id) === 'string' && parent.id.length > 0)
            this.id = parent.id + '/' + this.name;
        else
            this.id = this.name;
        this.selected = false;

        this.children = {};
        if(typeof(childDefinitions) === 'object') {
            let childKeys = Object.keys(childDefinitions);
            for(var k=0; k<childKeys.length; k++) {
                let child = new DirectoryElement(childKeys[k], childDefinitions[childKeys[k]], this, this.browser);
                this.children[child.id] = child;
            }   
        }
        this._setup_markup();
        browser.registerElement(this);
    }

    _setup_markup() {
        let self = this;

        this.markup = $('<div class="file-directory-entry" id="'+this.id+'"></div>');

        this.nameDiv = $('<div class="file-directory-entry-name">' + this.name + '</div>');
        this.nameDiv.on('click', function() {
            self._entry_clicked();
        });

        if(Object.keys(this.children).length > 0) {
            this.childDiv = $('<div class="file-directory-entry-children"></div>');
            for(var childKey in this.children) {
                this.childDiv.append(this.children[childKey].markup);
            }

            // add collapsible button
            let collapsible = $('<div class="file-directory-entry-collapsible">&#9654;</div>');   //TODO: symbol
            collapsible.on('click', function() {
                let rotationFactor = (self.childDiv.is(':visible') ? 0 : 90);
                self.childDiv.slideToggle();
                $(this).css({ 'transform': 'rotate(' + rotationFactor + 'deg)'})
                        .css({ 'WebkitTransform': 'rotate(' + rotationFactor + 'deg)'})
                        .css({ '-moz-transform': 'rotate(' + rotationFactor + 'deg)'});
            });
            this.markup.append(collapsible);
            this.markup.append(this.nameDiv);
            this.markup.append(this.childDiv);
        } else {
            this.markup.append(this.nameDiv);
        }
    }

    _entry_clicked() {
        this.setSelected(!this.selected);
        this.browser.entrySelected(this, this.selected);
    }

    childSelected(selected) {
        if(selected) {
            // highlight this entry in a dim color
            this.nameDiv.addClass('file-directory-highlighted-weak');
            this.nameDiv.removeClass('file-directory-highlighted');
            this.selected = false;
        } else {
            this.nameDiv.removeClass('file-directory-highlighted-weak');
        }
        this.parent.childSelected(selected);
    }

    setSelected(selected) {
        this.selected = selected;
        if(this.selected) {
            this.nameDiv.addClass('file-directory-highlighted');
        } else {
            this.nameDiv.removeClass('file-directory-highlighted');
        }
        this.parent.childSelected(this.selected);
    }
}


class DirectoryBrowser {

    constructor(domElement, tree) {
        this.domElement = $(domElement);

        this.elements = {};
        this.selectedElement = null;

        this.id = '';

        this.setTree(tree);

        this.callbacks = {
            'select': {},
            'deselect': {}
        };
    }

    _setup_markup() {
        this.domElement.empty();
        this.elements = {};
        this.selectedElement = null;

        if(this.tree === undefined || typeof(this.tree) !== 'object') return;

        for(var childKey in this.tree) {
            let child = new DirectoryElement(childKey, this.tree[childKey], this, this);
            this.domElement.append(child.markup);
        }
    }

    setTree(tree) {
        this.tree = tree;
        this._setup_markup();
    }

    registerElement(element) {
        this.elements[element.id] = element;
    }

    childSelected(selected) {
        return;
    }

    entrySelected(child, selected) {
        if(this.selectedElement !== null) {
            this.elements[this.selectedElement].setSelected(false);
            this.selectedElement = null;
        }
        if(child !== undefined && child !== null) {
            if(selected) {
                this.selectedElement = child.id;
                this._fireEvent('select', child);
            } else {
                this._fireEvent('deselect', child);
            }
        }
    }

    getSelected() {
        if(this.selectedElement === null) return null;
        else return this.elements[this.selectedElement];
    }

    setSelected(id) {
        if(this.elements.hasOwnProperty(id)) {
            this.entrySelected(this.elements[id], true);
        } else {
            this.entrySelected(null);
        }
    }

    _fireEvent(event, item) {
        if(this.callbacks.hasOwnProperty(event)) {
            for(var key in this.callbacks[event]) {
                this.callbacks[event][key](item);
            }
        }
    }

    on(event, callback) {
        this.callbacks[event][callback] = callback;
    }

    off(event) {
        if(this.callbacks.hasOwnProperty(event)) {
            this.callbacks[event] = {};
        }
    }
}