/**
 * Displays a progress bar and current job (note: not (Celery) task) name next
 * to it in the footer of the interface. Used for jobs that run on the server
 * side, but not as a Celery task.
 *
 * 2021 Benjamin Kellenberger
 */

class JobIndicator {

    constructor(containerDiv) {
        this.containerDiv = containerDiv;
        this.jobs = {};
        this._setup_markup();
    }

    _setup_markup() {
        this.containerDiv.hide();
        this.jobNameField = $('<span></span>');
        this.containerDiv.append(this.jobNameField);

        this.pbar = new ProgressBar(true, 0, 0, true);
        this.containerDiv.append(this.pbar.getMarkup());
    }

    _update_markup() {
        let keys = Object.keys(this.jobs);
        if(keys.length) {
            if(keys.length > 1) {
                this.jobNameField.html(keys.length.toString() + ' jobs');
            } else {
                let jobName = this.jobs[keys[0]];
                this.jobNameField.html(jobName);
            }
            this.containerDiv.show();
        } else {
            this.containerDiv.hide();
        }
    }

    addJob(id, message) {
        this.jobs[id] = message;
        this._update_markup();
    }

    removeJob(id) {
        if(this.jobs.hasOwnProperty(id)) {
            delete this.jobs[id];
        }
        this._update_markup();
    }
}