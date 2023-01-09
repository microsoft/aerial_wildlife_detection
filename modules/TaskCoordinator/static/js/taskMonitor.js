/**
 * TaskMonitor, for issuing, aborting, and keeping track of all non-AI tasks.
 *
 * 2021-23 Benjamin Kellenberger
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
    'AIController.delete_model_states': 'Delete model states',
    'AIController.get_model_training_statistics': 'Get model training statistics',
    'AIController.duplicate_model_state': 'Duplicate model state',
    'AIController.get_training_images': 'Collect training images',
    'AIController.get_inference_images': 'Collect inference images',
    'AIWorker.call_update_model': 'Update model',
    'AIWorker.call_train': 'Train model',
    'AIWorker.call_average_model_states': 'Combine model states',
    'AIWorker.call_inference': 'Inference',
    'DataAdministration.verify_images': 'Verify images',
    'DataAdministration.list_images': 'List images',
    'DataAdministration.scan_for_images': 'Scan file server for untracked images',
    'DataAdministration.add_existing_images': 'Add existing images to project',
    'DataAdministration.remove_images': 'Remove images',
    'DataAdministration.request_annotations': 'Request annotation download',
    'DataAdministration.delete_project': 'Delete project',
    'DataAdministration.verify_images': 'Verify image integrity',
    'ModelMarketplace.shareModel': 'Share model',
    'ModelMarketplace.importModelDatabase': 'Import model',
    'ModelMarketplace.importModelURI': 'Import model from Web',
    'ModelMarketplace.requestModelDownload': 'Request model download',
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
            let showInfoContainer = $('<div style="padding:2px"></div>');

            if(this.showAdminFunctionalities) {
                // abort button
                let revokeTaskButton = $('<button class="btn btn-sm btn-danger">Abort</button>');
                revokeTaskButton.click(function() {
                    self.revokeTask();
                });
                this.statusIcon.append(revokeTaskButton);

                // delete button
                this.deleteTaskButton = $('<button class="btn btn-sm btn-danger task-info-button" style="display:none">Delete</button>');
                this.deleteTaskButton.on('click', function() {
                    self.deleteTask();
                });
                showInfoContainer.append(this.deleteTaskButton);
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
                    statusText = '';
                    if(updates['name'].toLowerCase() === 'celery.chord') {
                        // special case for Chords that don't seem to finish even if children did
                        let successful = true;
                        for(var child in updates['children']) {
                            if(updates['children'][child]['successful'] !== true) {
                                successful = false;
                                break;
                            }
                        }
                        if(successful) {
                            this.hasFinished = true;
                        }
                    }
                    break;
                default:
                    statusText = statusID.toLowerCase();
            }
        }
        if(!this.hasFinished && !this.hasFailed && updates.hasOwnProperty('info') && updates['info'] !== null && updates['info'] !== undefined) {
            let info = updates['info'];
            this.done = parseFloat(info['done']);
            this.total = parseFloat(info['total']);
            let indefinite = (isNaN(this.done) || isNaN(this.total));
            this.pBar.set(true, this.done, this.total, indefinite);
            if(typeof(info['message']) === 'string') {
                statusText += ': ' + info['message'];
            }
            if(!indefinite) {
                statusText += ' ('+this.done+'/'+this.total+')';
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

                this.deleteTaskButton.show();
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
        if(this.taskFinished() || this.taskFailed()) {
            this.pBar.set(false);
        } else if(this.childTasks.length > 0) {
            if(numChildrenDone === this.childTasks.length) {
                this.hasFinished = true;
                this.pBar.set(false);
            } else if(!this.hasFinished && !this.hasFailed) {
                this.pBar.set(true, numChildrenDone, this.childTasks.length, false);
            }
        }

        // // time finished variable has precedence over all others
        // this.timeFinished = parseFloat(updates['time_finished']);      //TODO: show time required
        // if(!isNaN(this.timeFinished)) {
        //     this.hasFinished = true;
        // }
        if(this.hasFinished) {
            this._set(this.timeFinished, !this.hasFailed);
        }
    }

    revokeTask() {
        //TODO: rewrite for TaskCoordinator
        return;
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
        });
    }

    deleteTask() {
        //TODO: rewrite for TaskCoordinator
        return
        let self = this;
        return $.ajax({
            url: window.baseURL + 'deleteWorkflowHistory',
            method: 'POST',
            contentType: 'application/json; charset=utf-8',
            dataType: 'json',
            data: JSON.stringify({
                workflow_id: self.id
            }),
            success: function(data) {
                if(data['status'] !== 0) {
                    let message = 'Workflow could not be deleted';
                    if(typeof(data['message']) === 'string') {
                        message += ' (message: "' + data['message'] + '")';
                    }
                    message += '.';
                    window.messager.addMessage(message, 'error', 0);
                } else {
                    self.markup.slideUp();
                    self.markup.detach();
                    //TODO: remove task from parent
                }
            },
            error: function(xhr, status, error) {
                console.error(error);
                window.messager.addMessage('Workflow with id "'+self.id+'" could not be deleted (message: "'+error+'").', 'error', 0);
            },
            statusCode: {
                401: function(xhr) {
                    return window.renewSessionRequest(xhr, function() {
                        self.deleteTask();
                    });
                }
            }
        });
    }

    _set(timeFinished, successful) {
        /* 
            Forcefully sets task and all children to finished,
            and also to successful or failed, depending on
            attribute.
            Also hides progress bar of task and all children.
         */
        if(!isNaN(timeFinished)) {
            this.timeFinished = timeFinished;
        }
        if(successful === true) {
            this.successful = true;
        } else if(successful === false) {
            this.hasFailed = true;
        }
        this.pBar.set(false);
        for(var c=0; c<this.childTasks.length; c++) {
            this.childTasks[c]._set(timeFinished, successful);
            // if(successful === true) {
            //     this.childTasks[c].successful = true;
            // } else if(successful === false) {
            //     this.childTasks[c].hasFailed = true;
            // }
            // this.childTasks[c].hasFinished = true;
        }
    }

    taskFailed() {
        // check children first
        for(var c=0; c<this.childTasks.length; c++) {
            if(this.childTasks[c].taskFailed()) {
                this.hasFailed = true;
                this.pBar.set(false);
                break;
            }
        }

        // set children if task has failed and hide progress bar
        if(this.hasFailed) {
            this._set(this.timeFinished, false);
            this.pBar.set(false);
        }

        return this.hasFailed;
    }

    taskFinished() {
        if(this.isRootTask) {
            if(this.hasFinished) {
                // force children to show finished flag too
                this._set(this.timeFinished);
                this.pBar.set(false);

            } else {
                // check children
                for(var c=0; c<this.childTasks.length; c++) {
                    if(!this.childTasks[c].taskFinished()) {
                        return false;
                    }
                    // if(this.childTasks[c].taskSuccessful() === false) {
                    //     // error: entire task aborted
                    //     this.hasFinished = true;
                    // }
                    // this.childTasks[c].taskFinished();
                }
            }
        }

        // hide progress bar
        if(this.hasFinished)
            this.pBar.set(false);

        return this.hasFinished;
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

    getProgress() {
        if(this.taskFinished()) return 1.0;
        let progress = Math.max(0, Math.min(this.done / this.total, 1.0));
        if(isNaN(progress)) progress = 0.0;
        return progress;
    }

    showDetails(visible) {
        if(visible) this.childrenDiv.slideDown();
        else this.childrenDiv.slideUp();
    }
}


class TaskMonitor {

    constructor(domElement_main, domElement_footer, showAdminFunctionalities,
        queryInterval) {
            
        this.domElement_main = domElement_main;
        this.domElement_footer = domElement_footer;
        this.showAdminFunctionalities = showAdminFunctionalities;
        this.queryInterval = (!isNaN(parseFloat(queryInterval)) ? parseFloat(queryInterval) : 1000);

        this.tasks = {};

        this._setup_markup();
    }

    _setup_markup() {
        var self = this;
        this.domElement_main.empty();
        this.runningTasksContainer = $('<div class="task-list" id="running-tasks-list"></div>');
        this.domElement_main.append(this.runningTasksContainer);
        this.finishedTasksContainer = $('<div class="task-list" id="finished-tasks-list"></div>');
        let ftHead = $('<h4 class="task-list-header">Finished tasks</h4>');
        ftHead.click(function() {
            self.finishedTasksContainer.slideToggle();
        });
        let ftHeadContainer = $('<div></div>');
        ftHeadContainer.append(ftHead);
        // let deleteAll = $('<button class="btn btn-sm btn-danger">Delete all</button>');
        // deleteAll.on('click', function() {
        //     self.deleteAllWorkflows(false);
        // });
        // ftHeadContainer.append(deleteAll);
        this.domElement_main.append(ftHeadContainer);
        this.domElement_main.append(this.finishedTasksContainer);
    }

    _do_query() {
        //TODO
        let self = this;
        let queryURL = window.baseURL + 'pollStatus';
        return $.ajax({
            url: queryURL,
            method: 'GET',
            success: function(data) {
                // running tasks
                let totalProgress = 0;
                let targetProgress = 0;
                let tasks = data['status']['tasks'];
                if(tasks === undefined || tasks === null || (Array.isArray(tasks) && tasks.length === 0)) {
                    tasks = [];
                    // self.setQueryInterval(self.queryIntervals['idle'], false);
                } else {
                    // self.setQueryInterval(self.queryIntervals['active'], false);
                }
                let numActiveTasks = 0;
                for(var t=0; t<tasks.length; t++) {
                    let taskID = tasks[t]['id'];
                    let task = self.tasks[taskID];
                    if(task === undefined) {
                        // new task found
                        task = new Task(tasks[t], undefined, true, self.showAdminFunctionalities);
                        self.tasks[taskID] = task;
                        if(task.taskFinished() || task.taskFailed() || task.taskSuccessful()) {
                            self.finishedTasksContainer.append(task.markup);
                            task.showDetails(false);
                        } else {
                            numActiveTasks++;
                            self.runningTasksContainer.append(task.markup);
                        }
                    } else {
                        task.updateStatus(tasks[t]);
                        if(task.taskFinished() || task.taskFailed() || task.taskSuccessful()) {
                            if(task.markup.parent().attr('id') !== 'finished-tasks-list') {
                                task.markup.detach();
                                self.finishedTasksContainer.prepend(task.markup);
                                task.showDetails(false);
                            }
                        } else {
                            numActiveTasks++;
                        }
                    }

                    let taskProgress = task.getProgress();
                    if(taskProgress > 0) {
                        targetProgress += 1;
                        totalProgress += task.getProgress();
                    }
                }
                if(numActiveTasks > 0) {
                    // self.setQueryInterval(self.queryIntervals['active'], false);
                } else {
                    // self.setQueryInterval(self.queryIntervals['idle'], false);
                    totalProgress = 0;
                }
                let taskSuffix = (numActiveTasks === 1 ? ' task' : ' tasks');
                self._set_footer_progress('tasks', true, totalProgress, targetProgress, (targetProgress<=0 && numActiveTasks>0), numActiveTasks + taskSuffix);

                
                // auto-train
                try {
                    if(data['status'].hasOwnProperty('project')) {
                        let placeholderText = '(auto-training disabled)';
                        let autoTrainingEnabled = data['status']['project']['ai_auto_training_enabled'];
                        let numAnnotated = data['status']['project']['num_annotated'];
                        let numNext = data['status']['project']['num_next_training'];
                        if(autoTrainingEnabled && numNext > 0) {
                            if(numAnnotated >= numNext) {
                                placeholderText = 'in queue...';
                            } else {
                                placeholderText = numAnnotated+'/'+numNext;
                            }
                        }
                    }
                } catch {
                }
            },
            error: function(xhr, status, error) {
                //TODO
                console.error(error);
                self.setQueryInterval(self.queryIntervals['error'], false);
                return window.renewSessionRequest(xhr, function() {
                    self._query_auto();
                });
            }
        });
    }

    _query_auto() {
        let self = this;
        let promise = this._do_query();
        return promise.done(function() {
            if(self.queryInterval > 0) {
                setTimeout(function() {
                    self._query_auto();
                }, self.queryInterval);
            }
        });
    }

    startQuerying() {
        this._query_auto();
    }

    queryNow() {
        return this._do_query();
    }
}