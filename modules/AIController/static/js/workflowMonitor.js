/**
 * WorkflowMonitor, for issuing, aborting, and keeping track
 * of AI model training and inference workflows for a specific
 * project.
 * 
 * 2020 Benjamin Kellenberger
 */

$(document).ready(function() {
    if(typeof(ProgressBar) === 'undefined') {
        $.getScript('/static/general/js/progressbar.js');
    }

    if(window.baseURL === undefined) {
        window.baseURL = '';
    }
});



var TASK_NAME_MAP = {
    'AIController.get_training_images': 'Collect training images',
    'AIController.get_inference_images': 'Collect inference images',
    'AIWorker.call_update_model': 'Update model',
    'AIWorker.call_train': 'Train model',
    'AIWorker.call_average_model_states': 'Combine model states',
    'AIWorker.call_inference': 'Inference',
    'celery.chord': 'Composite task',
    'celery.chain': 'Task sequence'
}


class Task {

    constructor(meta, number, isRootTask, showAdminFunctionalities) {
        this.id = meta['id'];
        this.showAdminFunctionalities = showAdminFunctionalities;
        this.hasFinished = false;
        this.hasFailed = false;
        this.launchedBy = meta['launched_by'];
        if(typeof(this.launchedBy) !== 'string') this.launchedBy = '(automatic)';
        this.abortedBy = meta['aborted_by'];
        if(typeof(this.abortedBy) !== 'string') this.abortedBy = ''
        else this.hasFinished = true;
        this.messages = meta['messages'];
        if(typeof(this.messages) !== 'string') this.messages = '';
        this.timeCreated = parseFloat(meta['time_created']);
        this.taskName = TASK_NAME_MAP[meta['name']];
        if(this.taskName === undefined || this.taskName === null) {
            if(!isNaN(this.timeCreated)) {
                this.taskName = new Date(this.timeCreated * 1000).toLocaleString();
            } else {
                this.taskName = 'Unknown task';
            }
        }
        this.number = number;
        this.isRootTask = isRootTask;
        
        this.childTasks = [];
        this.childTasksMap = {};
        if(meta.hasOwnProperty('children')) {
            for(var c=0; c<meta['children'].length; c++) {
                let childMeta = meta['children'][c];
                let childTask = new Task(childMeta, (c+1), false, this.showAdminFunctionalities);
                this.childTasksMap[childMeta['id']] = this.childTasks.length;
                this.childTasks.push(childTask);
            }
        }
        this.successful = undefined;    // if true or false, task has completed (or failed)

        this._setup_markup();
        this.updateStatus(meta);
    }

    _setup_markup() {
        var self = this;
        let divClass = (this.isRootTask? 'task' : 'subtask');
        this.markup = $('<div class="'+divClass+'"></div>');
        let headerDiv = $('<div class="task-header"></div>');
        let subHeaderDiv = $('<div class="task-sub-header"></div>');
        if(this.number !== undefined) {
            subHeaderDiv.append($('<span class="task-number">'+this.number+'</span>'));
        }
        subHeaderDiv.append($('<span class="task-type">'+this.taskName+'</span>'));
        this.statusIndicator = $('<span>Querying...</span>');
        subHeaderDiv.append(this.statusIndicator);
        headerDiv.append(subHeaderDiv);
        this.statusIcon = $('<div class="task-status-icon"></div>');
        if(this.isRootTask) {
            if(this.showAdminFunctionalities) {
                // abort button
                let revokeTaskButton = $('<button class="btn btn-sm btn-danger">Abort</button>');
                revokeTaskButton.click(function() {
                    self.revokeTask();
                });
                this.statusIcon.append(revokeTaskButton);
            }
            this.markup.append(headerDiv);

            // additional task info //TODO: make dynamically updateable
            let infoPane = $('<div class="task-info-pane"></div>');
            infoPane.append($('<table class="task-info-table">' +
                '<tr><td>Launched by:</td><td>' + this.launchedBy + '</td></tr>' +
                '<tr><td>Aborted by:</td><td>' + this.abortedBy + '</td></tr>' +
                '<tr><td>Messages:</td><td>' + this.messages + '</td></tr></table>'));
            let showInfoButton = $('<button class="btn btn-sm btn-info task-info-button">Info</button>');
            showInfoButton.click(function() {
                infoPane.slideToggle();
            });
            let showInfoContainer = $('<div style="padding:2px"></div>');
            showInfoContainer.append(showInfoButton);
            headerDiv.append(showInfoContainer);
            this.markup.append(infoPane);

            headerDiv.append(this.statusIcon);
        } else {
            this.markup.append(headerDiv);
        }

        let progressDiv = $('<div class="task-progress-container"></div>');
        this.pBar = new ProgressBar(false, 0, 0, true);
        progressDiv.append(this.pBar.markup);
        this.markup.append(progressDiv);

        this.childrenDiv = $('<div class="task-children-list"></div>');
        for(var c=0; c<this.childTasks.length; c++) {
            this.childrenDiv.append(this.childTasks[c].markup);
        }
        if(this.childTasks.length > 0) {
            //TODO: add triangle
            subHeaderDiv.click(function() {
                self.childrenDiv.slideToggle();
            });
            subHeaderDiv.css('cursor', 'pointer');
            this.markup.append(this.childrenDiv);
        }
    }

    updateStatus(updates) {
        //TODO: update GUI
        this.abortedBy = updates['aborted_by'];
        if(typeof(this.abortedBy) !== 'string') this.abortedBy = ''
        else this.hasFinished = true;
        this.timeFinished = parseFloat(updates['time_finished']);      //TODO: show time required
        if(!isNaN(this.timeFinished)) {
            this.hasFinished = true;
        }
        this.messages = updates['messages'];
        if(typeof(this.messages) !== 'string') this.messages = '';

        let statusText = '';
        if(typeof(updates['status']) === 'string') {
            let statusID = updates['status'].toUpperCase();
            switch(statusID) {
                case 'SUCCESS':
                    statusText = 'completed.';
                    this.hasFinished = true;
                    break;
                case 'FAILURE':
                    statusText = 'failed';
                    if(typeof(updates['messages']) === 'string') {
                        statusText += ' ' + updates['messages'];
                    }
                    this.hasFailed = true;
                    this.hasFinished = true;
                    this.taskFailed();
                    break;
                case 'PROGRESS':
                    statusText = 'working';
                    break;
                case 'PENDING':
                    statusText = '';    //TODO
                    break;
                default:
                    statusText = statusID.toLowerCase();
            }
        }
        if(!this.hasFinished && !this.hasFailed && updates.hasOwnProperty('info') && updates['info'] !== null && updates['info'] !== undefined) {
            let info = updates['info'];
            let done = parseFloat(info['done']);
            let total = parseFloat(info['total']);
            let indefinite = (isNaN(done) || isNaN(total));
            this.pBar.set(true, done, total, indefinite);
            if(typeof(info['message']) === 'string') {
                statusText += ': ' + info['message'];
            }
            if(!indefinite) {
                statusText += ' ('+done+'/'+total+')';
            }
        }
        if(updates.hasOwnProperty('succeeded')) {
            // task completed
            this.successful = updates['succeeded'];
            if(!this.hasFinished) this.hasFinished = (typeof(this.successful) === 'boolean');
            if(this.hasFinished || this.taskFailed()) {
                this.statusIcon.empty();
                if(updates['succeeded'] && !this.taskFailed()) {
                    this.statusIcon.append($('<img src="/static/general/img/success.svg" />'));
                } else {
                    this.statusIcon.append($('<img src="/static/general/img/error.svg" />'));
                }
            }
        }

        let numChildrenDone = 0;
        if(updates.hasOwnProperty('children') && Object.keys(updates['children']).length > 0) {
            for(var child in updates['children']) {
                let childID = updates['children'][child]['id'];
                if(this.childTasksMap[childID] !== undefined) {
                    this.childTasks[this.childTasksMap[childID]].updateStatus(updates['children'][child]);
                    if(this.childTasks[this.childTasksMap[childID]].taskFinished()) {
                        numChildrenDone++;
                    }
                }
            }
        }
        this.statusIndicator.html(statusText);
        if(this.hasFinished || this.taskFailed()) {
            this.pBar.set(false);
        } else if(this.childTasks.length > 0) {
            if(numChildrenDone === this.childTasks.length) {
                this.hasFinished = true;
                this.pBar.set(false);
            } else if(!this.hasFinished && !this.hasFailed) {
                this.pBar.set(true, numChildrenDone, this.childTasks.length, false);
            }
        }
    }

    revokeTask() {
        //TODO
        var self = this;
        return $.ajax({
            url: window.baseURL + 'abortWorkflow',
            method: 'POST',
            contentType: 'application/json; charset=utf-8',
            dataType: 'json',
            data: JSON.stringify({
                taskID: self.id
            }),
            success: function(data) {
                let message = 'Workflow aborted.';
                let success = undefined;
                let duration = undefined;
                if(data['status'] !== 0) {
                    success = 'error';
                    duration = 0;
                    message = 'Workflow could not be aborted';
                    if(typeof(data['message']) === 'string') {
                        message += ' (message: "' + data['message'] + '")';
                    }
                    message += '.';
                } else {
                    self.hasFailed = true;
                    self.taskFailed();
                }
                window.messager.addMessage(message, success, duration);
            },
            error: function(xhr, status, error) {
                console.error(error);
                window.messager.addMessage('Workflow with id "'+self.id+'" could not be revoked (message: "'+error+'").', 'error', 0);
            },
            statusCode: {
                401: function(xhr) {
                    return window.renewSessionRequest(xhr, function() {
                        self.revokeTask();
                    });
                }
            }
        })
    }

    taskFailed() {
        if(this.hasFailed) {
            // assign all children to failed as well
            for(var c=0; c<this.childTasks.length; c++) {
                this.childTasks[c].hasFailed = true;
            }
        } else {
            // also assign failed if any of the children have failed
            for(var c=0; c<this.childTasks.length; c++) {
                if(this.childTasks[c].taskFailed()) {
                    this.hasFailed = true;
                    return true;
                }
            }
        }

        // hide progress bar
        if(this.hasFailed)
            this.pBar.set(false);

        return this.hasFailed;
    }

    taskFinished() {
        let taskFinished = this.hasFinished || this.taskFailed();

        // check children
        if(this.isRootTask) {
            for(var c=0; c<this.childTasks.length; c++) {
                if(this.childTasks[c].taskSuccessful() === false) {
                    // error: entire task aborted
                    this.taskFinished = true;
                }
                this.childTasks[c].taskFinished();
            }
        }
        this.hasFinished = taskFinished;

        // hide progress bar
        if(this.hasFinished)
            this.pBar.set(false);

        return taskFinished;
    }

    taskSuccessful() {
        let successful = true;
        
        if(this.isRootTask) {
            // the root task has finished if all its children have done so
            for(var c=0; c<this.childTasks.length; c++) {
                let childSuccessful = this.childTasks[c].taskSuccessful();
                if(childSuccessful === undefined) {
                    successful = undefined;
                } else if(!childSuccessful) {
                    return false;
                }
            }
            return successful;
        } else {
            return this.successful && successful;
        }
    }
}


class WorkflowMonitor {

    constructor(domElement_main, domElement_footer, showAdminFunctionalities, queryProject,
        queryInterval_active, queryInterval_idle, queryInterval_error) {
            
        this.domElement_main = domElement_main;
        this.domElement_footer = domElement_footer;
        this.showAdminFunctionalities = showAdminFunctionalities;
        this.queryProject = queryProject;       // tasks are always queried
        this.queryIntervals = {
            'idle': (!isNaN(parseFloat(queryInterval_idle)) ? parseFloat(queryInterval_idle) : 10000),
            'active': (!isNaN(parseFloat(queryInterval_active)) ? parseFloat(queryInterval_active) : 1000),
            'error': (!isNaN(parseFloat(queryInterval_error)) ? parseFloat(queryInterval_error) : 10000)
        };
        this.queryInterval = queryInterval_active;

        this.tasks = {};

        this._setup_markup();
        this.setQueryInterval(this.queryIntervals['active'], true);
    }

    _setup_markup() {
        var self = this;
        this.domElement_main.empty();
        //TODO: make prettier; add triangles
        this.runningTasksContainer = $('<div class="task-list" id="running-tasks-list"></div>');
        let rwHead = $('<h3 class="task-list-header">Running workflows</h3>');
        rwHead.click(function() {
            self.runningTasksContainer.slideToggle();
        })
        this.domElement_main.append(rwHead);
        this.domElement_main.append(this.runningTasksContainer);
        this.finishedTasksContainer = $('<div class="task-list" id="finished-tasks-list"></div>');
        let fwHead = $('<h3 class="task-list-header">Finished workflows</h3>');
        fwHead.click(function() {
            self.finishedTasksContainer.slideToggle();
        })
        this.domElement_main.append(fwHead);
        this.domElement_main.append(this.finishedTasksContainer);

        // footer
        if(typeof(this.domElement_footer) === 'object') {
            this.domElement_footer = $(this.domElement_footer);
            let footerContainer = $('<table class="footer-container"></table>');

            // progress bar for running tasks
            this.runningTasksPbar = new ProgressBar(true, 0, 0, false);
            let rtDiv = $('<tr class="footer-pbar-inline"></tr>');
            rtDiv.append($('<td class="footer-span">running tasks:</td>'));
            let rtCell = $('<td class="footer-pbar-cell"></td>');
            rtCell.append(this.runningTasksPbar.getMarkup());
            this.runningTasksPlaceholder = $('<span>(none)</span>');
            rtCell.append(this.runningTasksPlaceholder);
            rtDiv.append(rtCell);
            this.runningTasksCounter = $('<td></td>');
            rtDiv.append(this.runningTasksCounter);
            footerContainer.append(rtDiv);

            // progress bar for # annotated images until auto re-training
            this.autoTrainPbar = new ProgressBar(true, 0, 0, false);
            let atDiv = $('<tr class="footer-pbar-inline"></tr>');
            atDiv.append($('<td class="footer-span">images until re-train:</td>'));
            let atCell = $('<td class="footer-pbar-cell"></td>');
            atCell.append(this.autoTrainPbar.getMarkup());
            this.autoTrainPlaceholder = $('<span>(auto-training disabled)</span>');
            atCell.append(this.autoTrainPlaceholder);
            atDiv.append(atCell);
            this.autoTrainCounter = $('<td></td>');
            atDiv.append(this.autoTrainCounter);
            footerContainer.append(atDiv);

            this.domElement_footer.append(footerContainer);
        }
    }

    _do_query(nudgeWatchdog) {
        let self = this;
        let queryURL = window.baseURL + 'status?tasks=true';
        if(this.queryProject) queryURL += '&project=true';
        if(nudgeWatchdog) queryURL += '&nudge_watchdog=true';
        return $.ajax({
            url: queryURL,
            method: 'GET',
            success: function(data) {

                // running tasks
                let tasks = data['status']['tasks'];
                if(tasks === undefined || tasks === null || (Array.isArray(tasks) && tasks.length === 0)) {
                    tasks = [];
                    self.setQueryInterval(self.queryIntervals['idle'], false);
                } else {
                    self.setQueryInterval(self.queryIntervals['active'], false);
                }
                let numActiveTasks = 0;
                for(var t=0; t<tasks.length; t++) {
                    let taskID = tasks[t]['id'];
                    let task = self.tasks[taskID];
                    if(task === undefined) {
                        // new task found
                        task = new Task(tasks[t], undefined, true, self.showAdminFunctionalities);
                        self.tasks[taskID] = task;
                        if(task.taskFinished()) {
                            self.finishedTasksContainer.append(task.markup);
                        } else {
                            numActiveTasks++;
                            self.runningTasksContainer.append(task.markup);
                        }
                    } else {
                        task.updateStatus(tasks[t]);
                        if(task.taskFinished()) {
                            if(task.markup.parent().attr('id') !== 'finished-tasks-list') {
                                task.markup.detach();
                                self.finishedTasksContainer.prepend(task.markup);
                            }
                        } else {
                            numActiveTasks++;
                        }
                    }
                }
                if(numActiveTasks > 0) {
                    self.setQueryInterval(self.queryIntervals['active'], false);
                } else {
                    self.setQueryInterval(self.queryIntervals['idle'], false);
                }

                // auto-train
                try {
                    if(data['status'].hasOwnProperty('project')) {
                        let numAnnotated = data['status']['project']['num_annotated'];
                        let numNext = data['status']['project']['num_next_training'];
                        self._set_footer_progress('autotrain', numAnnotated, numNext, false);
                    }
                } catch {
                    self._set_footer_progress('autotrain', null);
                }
            },
            error: function(xhr, status, error) {
                //TODO
                console.error(error);
                self.setQueryInterval(self.queryIntervals['error'], false);
                return window.renewSessionRequest(xhr, function() {
                    self._query_workflows(nudgeWatchdog);
                });
            }
        });
    }

    _query_workflows(nudgeWatchdog) {
        let self = this;
        let promise = this._do_query(nudgeWatchdog);
        promise.done(function() {
            if(self.queryInterval > 0) {
                setTimeout(function() {
                    self._query_workflows(nudgeWatchdog);
                }, self.queryInterval);
            }
        });
    }

    _set_footer_progress(target, current, max, indefinite) {
        let pBar = null;
        let placeholder = null;
        let counterElement = null;
        if(target === 'tasks') {
            pBar = this.runningTasksPbar;
            placeholder = this.runningTasksPlaceholder;
            counterElement = this.runningTasksCounter;
        } else {
            pBar = this.autoTrainPbar;
            placeholder = this.autoTrainPlaceholder;
            counterElement = this.autoTrainPlaceholder;
        }
        if(typeof(pBar) !== 'object') return;
        
        if(typeof(current) !== 'number' && !indefinite) {
            pBar.set(true, 0, 0, false);
            placeholder.show();
            counterElement.hide();
        } else {
            pBar.set(true, current, max, indefinite);
            placeholder.hide();
            counterElement.html(current + '/' + max);
            counterElement.show();
        }
    }

    setQueryInterval(interval, startQuerying) {
        if(typeof(interval) === 'string' && this.queryIntervals.hasOwnProperty(interval)) {
            this.queryInterval = this.queryIntervals[interval];
        } else if(!isNaN(parseFloat(interval))) {
            this.queryInterval = interval;
        }
        if(startQuerying && this.queryInterval > 100) {
            this._query_workflows();
        }
    }

    startQuerying() {
        this.setQueryInterval('active', true);
    }

    queryNow(nudgeWatchdog) {
        return this._do_query(nudgeWatchdog);
    }
}