/**
 * Displays a progress bar and current task (operation) name next to it in the
 * footer of the interface. Used for longer-running tasks.
 * 
 * 2021 Benjamin Kellenberger
 */

class TaskMonitor {

    constructor(containerDiv) {
        this.containerDiv = containerDiv;
        this.tasks = {};
        this._setup_markup();
    }

    _setup_markup() {
        this.containerDiv.hide();
        this.taskNameField = $('<span></span>');
        this.containerDiv.append(this.taskNameField);

        this.pbar = new ProgressBar(true, 0, 0, true);
        this.containerDiv.append(this.pbar.getMarkup());
    }

    _update_markup() {
        let keys = Object.keys(this.tasks);
        if(keys.length) {
            if(keys.length > 1) {
                this.taskNameField.html(keys.length.toString() + ' tasks');
            } else {
                let taskName = this.tasks[keys[0]];
                this.taskNameField.html(taskName);
            }
            this.containerDiv.show();
        } else {
            this.containerDiv.hide();
        }
    }

    addTask(id, message) {
        this.tasks[id] = message;
        this._update_markup();
    }

    removeTask(id) {
        if(this.tasks.hasOwnProperty(id)) {
            delete this.tasks[id];
        }
        this._update_markup();
    }
}