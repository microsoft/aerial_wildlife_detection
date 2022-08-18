/**
 * Utility that displays an interface allowing users to manually
 * establish a mapping between AI model state-provided label
 * classes and those present in the project.
 * 
 * 2021-22 Benjamin Kellenberger
 */

class ModelLabelclassMapper {

    constructor(project, modelState) {
        this.project = project;

        /**
         * Model state:
         * - if dict with details, this comes from the Model Marketplace
         *   ("new" model)
         * - if null or undefined, model is from project (will load latest model state info)
         */
        this.modelState = modelState;
        if(this.modelState === undefined) this.modelState = null;

        /**
         * Flag for modification of dynamic model adaptation to new classes.
         * If last state of model had it activated, it cannot be disabled
         * unless a new model state is imported or old ones are deleted.
         */
        this.modelLibrary_supported = false;        // whether model library supports dynamic class adaptation
        this.modelAdaptation_enabled = false;       // whether model can incorporate new label classes
        this.modelAdaptation_changeable = false;    // whether behavior can be changed or not
    }

    _load_model_state_info() {
        if(this.modelState === null) {
            let self = this;

            // load project model state
            return $.ajax({
                url: window.baseURL + 'listModelStates?latest_only=1',
                success: function(data) {
                    try {
                        self.modelState = data['modelStates'][0];

                        // re-assign ID as per Model Marketplace origin
                        self.modelState['marketplace_id'] = self.modelState['marketplace_info']['id'];

                    } catch {
                        self.modelState = null;
                        self.labelclasses_model = {};
                    }
                }
            }).then(function() {
                // load label classes through Model Marketplace info
                if(self.modelState === null) return $.Deferred().resolve().promise();

                let marketplaceID = self.modelState['marketplace_id'];

                return $.ajax({
                    url: window.baseURL + 'getModelsMarketplace?model_ids=' + marketplaceID,
                    success: function(data) {
                        try {
                            self.modelState['labelclasses'] = data['modelStates'][marketplaceID]['labelclasses'];
                            self.labelclasses_model = self._parse_labelclasses(self.modelState['labelclasses']);
                        } catch {
                            self.modelState = null;
                            self.labelclasses_model = {};
                        }
                    }
                });
            });

        } else {
            // new model from Marketplace
            this.modelState['marketplace_id'] = this.modelState['id'];
            this.labelclasses_model = this._parse_labelclasses(this.modelState['labelclasses']);
            return $.Deferred().resolve().promise();
        }
    }

    _load_project_labelclasses() {
        let self = this;

        return $.ajax({
            url: window.baseURL + 'getClassDefinitions',
            method: 'GET',
            success: function(data) {
                self.labelclasses_project = self._parse_labelclasses(data['classes']['entries']);
            }
            //TODO: error, code
        })
    }

    _parse_labelclasses(labelclasses) {
        // iterates through labelclass definitions and flattens them into a list
        if(typeof(labelclasses) === 'string') {
            labelclasses = JSON.parse(labelclasses);
        }
        let output = {};
        function _parse_recursive(entry, id) {
            if(Array.isArray(entry)) {
                for(var e=0; e<entry.length; e++) {
                    _parse_recursive(entry[e], null);
                }
            } else if(typeof(entry) === 'string') {
                let entryID = (id===null ? entry : id);
                output[entryID] = entry;
            } else {
                if(entry.hasOwnProperty('entries')) {
                    // nested group
                    _parse_recursive(entry['entries'], null);
                } else if(entry.hasOwnProperty('name')) {
                    let entryID = (entry.hasOwnProperty('id') ? entry['id'] :
                        (id === null ? entry['name'] : id));
                    output[entryID] = entry['name'];
                } else {
                    let keys = Object.keys(entry);
                    for(var k=0; k<keys.length; k++) {
                        let key = keys[k];
                        let child = entry[key];
                        if(child.hasOwnProperty('id')) {
                            key = child['id'];
                        }
                        _parse_recursive(child, key);
                    }
                }
            }
        }
        _parse_recursive(labelclasses, null);
        return output;
    }

    _load_labelclass_map() {
        if(this.modelState === null || typeof(this.modelState) !== 'object') {
            this.labelclass_map = {};
            return $.Deferred().resolve().promise();
        }

        let self = this;
        self.labelclass_map = {};

        // initialize from model state classes
        for(var lcID in self.labelclasses_model) {
            self.labelclass_map[lcID] = null;
        }

        let data = '';
        if(self.modelState.hasOwnProperty('marketplace_id')) {
            data = '?modelID='+self.modelState['marketplace_id'];
        }

        return $.ajax({
            url: window.baseURL + 'getModelClassMapping' + data,
            method: 'GET',
            success: function(data) {
                data = data['response'];
                if(data.hasOwnProperty(self.modelState['marketplace_id'])) {
                    data = data[self.modelState['marketplace_id']];
                    for(var d=0; d<data.length; d++) {
                        let sourceID = data[d][0];
                        // let sourceName = data[d][1]; // not needed for mapping
                        let targetID = data[d][2];
                        self.labelclass_map[sourceID] = targetID;
                    }
                }
            }
        });
    }

    _get_labelclass_data() {
        let modelID = this.modelState['marketplace_id'];
        let mapping = [];
        for(var key in this.labelclass_map) {
            let sourceClassName = this.labelclasses_model[key];
            let targetClassID = this.labelclass_map[key];
            mapping.push([key, sourceClassName, targetClassID]);
        }
        let data = {};
        data['mapping'] = {};
        data['mapping'][modelID] = mapping;

        return data;
    }

    _save_labelclass_map() {
        if(this.modelState === null || typeof(this.modelState) !== 'object') {
            return $.Deferred().resolve().promise();
        }

        let data = this._get_labelclass_data();
        return $.ajax({
            url: window.baseURL + 'saveModelClassMapping',
            method: 'POST',
            contentType: 'application/json; charset=utf-8',
            dataType: 'json',
            data: JSON.stringify(data)
        });
    }

    _load_labelclass_autoadaptation_info() {
        /**
         * Loads info whether model is configured to automatically incorporate
         * new label classes (for existing model states) and whether the switch
         * to do so can be deactivated.
         * For new models (i.e., from Marketplace), the switch is off but enabled
         * by default.
         */

        let url = window.baseURL + 'getLabelclassAutoadaptInfo';
        if(this.modelState !== null && typeof(this.modelState) === 'object') {
            url += '?model_id=' + this.modelState['id'];
        }

        let self = this;
        return $.ajax({
            url: url,
            method: 'GET',
            success: function(data) {
                try {
                    self.modelLibrary_supported = data['message']['model_lib'];
                    self.modelAdaptation_enabled = (data['message']['project'] || data['message']['model']) && self.modelLibrary_supported;
                    self.modelAdaptation_changeable = (data['message']['model'] !== true);  // can only be deactivated if model state hasn't got it activated already
                } catch {
                    self.modelLibrary_supported = false;
                    self.modelAdaptation_enabled = false;
                    self.modelAdaptation_changeable = false;
                }
            }
        });
    }

    _get_labelclass_autoadaptation_info() {
        if(typeof(this.autoSwitch) !== 'object') return null;
        return this.autoSwitch.prop('checked');
    }

    _save_labelclass_autoadaptation_info() {
        if(typeof(this.autoSwitch) !== 'object') return $.Deferred().resolve().promise();
        let checked = this._get_labelclass_autoadaptation_info();
        return $.ajax({
            url: window.baseURL + 'saveLabelclassAutoadaptInfo?enabled=' + checked,
            method: 'POST'
        });
    }

    loadData() {
        // load project label classes and map, if present
        let self = this;
        return self._load_model_state_info().then(function() {
            return self._load_project_labelclasses().then(function() {
                return $.when(self._load_labelclass_map(), self._load_labelclass_autoadaptation_info());
            }).then(function() {
                return self._setup_markup();
            });
        });
    }

    saveData() {
        return $.when(this._save_labelclass_map(), this._save_labelclass_autoadaptation_info());
    }

    getData() {
        return {
            labelclass_map: this._get_labelclass_data(),
            auto_adapt: this._get_labelclass_autoadaptation_info()
        }
    }

    _update_disabled_selectors() {
        // make sure label classes can only be selected once
        let self = this;
        let targetVals = Object.values(self.labelclass_map);
        for(var s=0; s<self.selectors.length; s++) {
            let selID = self.selectors[s].attr('id').replace('selector__', '');
            self.selectors[s].find('option').each(function() {
                let val = $(this).val();
                $(this).prop('disabled',
                    targetVals.includes(val) &&
                    !(
                        self.labelclass_map.hasOwnProperty(selID) &&
                        self.labelclass_map[selID] == val
                    ) &&
                    val !== '$$add_new$$'
                );
            });
        }
    }

    _setup_markup() {
        let markup = $('<div></div>');

        if(this.modelState === null || typeof(this.modelState) !== 'object') {
            // no model state in project; show placeholder
            markup.append($('<div>No model state found in project.<br />You can import a model through the <a href="modelMarketplace">Model Marketplace</a>.</div>'));

        } else {

            let self = this;

            // model class search field
            let searchField = $('<input type="search" class="labelclass-filter-field" placeholder="filter model class" />');
            searchField.on('input', function() {
                self.filterModelClasses($(this).val());
            });
            markup.append(searchField);

            // auto-assign button
            let autoAssign = $('<button class="btn btn-sm btn-primary labelclass-auto-assign-button">auto-assign</button>');
            autoAssign.on('click', function() {
                self.autoAssignModelClasses();
            });
            markup.append(autoAssign);
            
            // add unassigned to project button
            let addUnassigned = $('<button class="btn btn-sm btn-primary labelclass-auto-add-button">add unassigned as new</button>');
            addUnassigned.on('click', function() {
                self.addUnassigned();
            });
            markup.append(addUnassigned);

            let mapTable = $('<table class="labelclass-mapper-table"><thead>' +
                '<tr><th>Model State</th><th>Project</th></tr>' +
                '</thead></table>');
            let mapTableBody = $('<tbody></tbody>');
            mapTable.append(mapTableBody);
            markup.append(mapTable);

            self.selectors = [];
            function _create_selector(labelclass_model) {
                let selector = $('<select id="selector__'+labelclass_model+'"></select>');
                selector.append($('<option value="">(unassigned)</option>'));
                selector.append($('<option value="$$add_new$$">(add to project)</option>'));
                for(var id in self.labelclasses_project) {
                    let name = self.labelclasses_project[id];
                    selector.append($('<option value="'+id+'">'+name+'</option>'));
                }
                if(self.labelclass_map.hasOwnProperty(labelclass_model)) {
                    selector.val(self.labelclass_map[labelclass_model]);
                }
                selector.on('input', function() {
                    // add to labelclass map
                    let sourceID = $(this).attr('id').replace('selector__', '');
                    let targetID = $(this).val();
                    if(targetID === '') {
                        // unassigned
                        delete self.labelclass_map[sourceID];
                    } else {
                        self.labelclass_map[sourceID] = targetID;
                    }

                    self._update_disabled_selectors();
                });
                self.selectors.push(selector);
                return selector;
            }
            self.tableEntries = [];
            for(var lcID in this.labelclasses_model) {
                let lcName = this.labelclasses_model[lcID];
                let entry = $('<tr id="lcmodel__'+lcID+'"><td>'+lcName+'</td></tr>');
                let targetCell = $('<td></td>');
                targetCell.append(_create_selector(lcID));
                entry.append(targetCell);
                mapTableBody.append(entry);
                self.tableEntries.push(entry);
            }
            self._update_disabled_selectors();
        

            // switch to enable or disable auto-adaptation
            let autoSwitchCheckbox = $('<div class="custom-control custom-switch"></div>');
            this.autoSwitch = $('<input type="checkbox" class="custom-control-input" id="auto-adapt-switch" />');
            this.autoSwitch.prop('checked', this.modelAdaptation_enabled && this.modelLibrary_supported);
            this.autoSwitch.prop('disabled', !this.modelAdaptation_changeable || !this.modelLibrary_supported);
            let autoSwitchLabel = $('<label class="custom-control-label" for="auto-adapt-switch">Enable model expansion towards new label classes</label>');
            autoSwitchCheckbox.append(this.autoSwitch);
            autoSwitchCheckbox.append(autoSwitchLabel);

            let autoSwitchDisclaimer = null;
            if(this.modelLibrary_supported) {
                autoSwitchDisclaimer = $('<p class="label-auto-switch-disclaimer">Activating this switch makes the model predict all project label classes, even if unassigned in the list above.<br />' +
                                                '<b>Notes:</b><ul><li>This involves adding new parameters to the model, which can reduce performance at the start. ' +
                                                'Best use it once you have a high number of labels for each class and can train a model for many epochs in the <a href="workflowDesigner">Workflow Designer</a>.</li>' +
                                                '<li>Once activated, this mode cannot be turned off anymore for any model states that are based on the current one.</li></ul></p>');
            } else {
                autoSwitchDisclaimer = $('<p>The model library currently used in this project does not support expansion to new label classes.</p>');
            }
            let autoSwitchContainer = $('<div style="margin-top:30px"></div>');
            autoSwitchContainer.append(autoSwitchCheckbox);
            autoSwitchContainer.append(autoSwitchDisclaimer);
            markup.append(autoSwitchContainer);
        }

        this.markup = markup;
    }

    getMarkup() {
        return this.markup;
    }

    filterModelClasses(keywords) {
        if(this.modelState === null || typeof(this.modelState) !== 'object') return;

        if(typeof(keywords) !== 'string') {
            keywords = '';
        }
        keywords = keywords.trim().split(' ');
        for(var k=0; k<keywords.length; k++) {
            keywords[k] = keywords[k].trim().toLowerCase();
        }

        for(var e=0; e<this.tableEntries.length; e++) {
            let entry = $(this.tableEntries[e]);
            let sourceClassName = $(entry.find('td')[0]).html().toLowerCase();
            let isMatch = true;
            for(var k=0; k<keywords.length; k++) {
                if(!sourceClassName.includes(keywords[k].toLowerCase())) isMatch = false;
            }
            if(isMatch) {
                entry.show();
            } else {
                entry.hide();
            }
        }
    }

    autoAssignModelClasses() {
        /**
         * Currently uses the Levenshtein distance to compare model and project
         * class names.
         */

        // NxM matrix of distances
        let lcModel = Object.keys(this.labelclasses_model);
        let lcProject = Object.keys(this.labelclasses_project);
        let distances = [];

        //TODO: mapreduce
        for(var lm=0; lm<lcModel.length; lm++) {
            for(var lp=0; lp<lcProject.length; lp++) {
                let dist = window.levDist(this.labelclasses_model[lcModel[lm]].trim().toLowerCase(),
                    this.labelclasses_project[lcProject[lp]].trim().toLowerCase());
                distances.push(dist);
            }
        }

        // assign in overall order
        let assigned_model = [];
        let assigned_proj = [];
        let order = window.argsort(distances);
        for(var o=0; o<order.length; o++) {
            let idx_model = Math.floor(order[o] / lcProject.length);
            let idx_proj = order[o] % lcProject.length;

            if(!assigned_model.includes(idx_model) && !assigned_proj.includes(idx_proj)) {
                // assign closest
                let selector = $('#selector__'+lcModel[idx_model]);
                selector.val(lcProject[idx_proj]);
                selector.trigger('input');
                assigned_model.push(idx_model);
                assigned_proj.push(idx_proj);
            }
        }
        this._update_disabled_selectors();
    }

    addUnassigned() {
        for(var lcID in this.labelclasses_model) {
            if(!this.labelclass_map.hasOwnProperty(lcID) || typeof(this.labelclass_map[lcID]) !== 'string') {
                this.labelclass_map[lcID] = '$$add_new$$';
                let selector = $('#selector__'+lcID);
                selector.val('$$add_new$$');
            }
        }
    }
}