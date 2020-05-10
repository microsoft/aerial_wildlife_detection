/*
    Functionality for monitoring and interacting connected AIWorkers
    from the client side.

    2019-20 Benjamin Kellenberger
*/


class AIWorker {
    /*
        Wrapper for worker status.
    */
    constructor(workerNumber, workerID, state) {
        this.workerNumber = workerNumber;
        this.workerID = workerID;
        this.state = state;
        this._setup_markup();
        this.update_state(state);
    }

    _setup_markup() {
        this.markup = $('<div class="ai-worker"></div>');
        this.markup.append('<h3>' + this.workerID + '</h3>');
        var detailsC = $('<table style="margin-left:20px;width:180px;font-size:14px;color:darkgray;"></table>');
        var activeTasksC = $('<tr><td>Active Tasks:</td></tr>');
        detailsC.append(activeTasksC);
        this.activeTasks = $('<td></td>');
        activeTasksC.append(this.activeTasks);
        var schedTasksC = $('<tr><td>Scheduled Tasks:</td></tr>');
        detailsC.append(schedTasksC);
        this.scheduledTasks = $('<td></td>');
        schedTasksC.append(this.scheduledTasks);
        this.markup.append(detailsC);
    }

    update_state(state) {
        if(state.hasOwnProperty('active_tasks')) {
            this.activeTasks.html(state['active_tasks'].length);
        } else {
            this.activeTasks.html('0');
        }
        if(state.hasOwnProperty('scheduled_tasks')) {
            this.scheduledTasks.html(state['scheduled_tasks'].length);
        } else {
            this.scheduledTasks.html('0');
        }
    }
}


class AIWorkerJob {
    /*
        Wrapper for individual jobs submitted to the worker(s).
    */
    constructor(jobNumber, jobID, state) {
        this.jobNumber = jobNumber;
        this.jobID = jobID;
        this.jobType = jobNumber + ' &mdash; ' + state['type'];


        //TODO: we don't parse the submitted date to avoid time zone shifts
        this.timeCreated = new Date();

        // milliseconds for a single time step
        this.averageTimeRequired = 1000;
        this.numPolls = 0;

        this._setup_markup();
        this._parse_state(state);
    }

    _parse_state(state) {
        //TODO: error handling?
        this.type = state['type'];
        // this.timeCreated = Date.parse(state['submitted']);
        this.status = state['status'];
        
        // status message
        try {
            var message = state['meta']['message'];
        } catch(err) {
            var message = 'running';
        }

        // progress
        if(this.status === 'PROGRESS') {
            this.total = parseInt(state['meta']['total']);
            this.current = Math.min(this.total, parseInt(state['meta']['done']));
            this.numPolls += 1;

            this.averageTimeRequired += this.timeElapsed / Math.max(1, this.current);
            this.timeRemaining = (this.total - this.current) * (this.averageTimeRequired / Math.max(1,this.numPolls));

            // update progress bar
            var newWidthPerc = 100*(this.current / this.total);
            this.progressBar.animate({
                'width': newWidthPerc + '%'
            }, 1000);

            // update message
            message += ' (' + this.current + '/' + this.total + ')';

        } else if(this.status === 'SUCCESS' || this.status === 'FAILURE') {
            // completed
            message = (this.status === 'SUCCESS' ? 'completed' : 'failed');
            if(state.hasOwnProperty('meta') && state['meta'] !== null && state['meta'].hasOwnProperty('message')) {
                message += ' (' + state['meta']['message'] + ')';
            }

            this.progressBar.parent().hide();

            // stop timers
            $(self.elapsedTime).hide();
            $(self.remainingTime).hide();
            clearInterval(this.timerHandle);

        } else {
            // no progress status; set indeterminate progress bar
            this.progressBar.css('width', '100%');
        }

        // status message
        this.statusDiv.html(message);


        // update time counter (markup will automatically update)
        this.timeElapsed = new Date() - this.timeCreated;
    }

    _setup_markup() {
        this.markup = $('<div class="ai-worker-job"></div>');

        // job type
        this.markup.append($('<h3 style="margin-left:12px;cursor:pointer">' + this.jobType + '</h3>'));

        // details container
        var detailsC = $('<div class="job-details"></div>');

        // status message
        var statusInd = $('<div></div>');
        statusInd.append($('<span>status: </span>'));
        this.statusDiv = $('<span></span>');
        statusInd.append(this.statusDiv);
        detailsC.append(statusInd);

        // progress indicator
        var prInd = $('<div></div>');
        var pbarWrapper = $('<div class="progressbar"></div>');
        prInd.append(pbarWrapper);
        this.progressBar = $('<div class="progressbar-filler progressbar-active"></div>');
        pbarWrapper.append(this.progressBar);

        // elapsed and remaining time
        var timeInd = $('<div class="time-indicator"></div>');
        this.elapsedTime = $('<div>00:00</div>');
        timeInd.append(this.elapsedTime);
        this.remainingTime = $('<div style="float:right">-00:00</div>');
        timeInd.append(this.remainingTime);
        prInd.append(timeInd);
        var self = this;
        this.timerHandle = setInterval(function() {
            self.timeElapsed += 1000;
            self.timeRemaining = Math.max(0, self.timeRemaining - 1000);
            $(self.elapsedTime).html(window.msToTime(self.timeElapsed));
            if(!isNaN(self.timeRemaining)) {
                $(self.remainingTime).html('-'+window.msToTime(self.timeRemaining));
                $(self.remainingTime).show();
            } else {
                $(self.remainingTime).hide();
            }
        }, 1000);

        detailsC.append(prInd);
        this.markup.append(detailsC);
    }

    update_state(state) {
        this._parse_state(state);
    }
}




class AIWorkerHandler {

    constructor(parentDiv) {
        this.parentDiv = parentDiv;
        this.workers = {};
        this.tasks = {};
        this._setup_markup();
        this._query_worker_status();
        this._query_task_status();
    }

    __toggle_panel() {
        if($('#ai-worker-panel').is(':visible')) {
            $('#ai-worker-panel').slideUp();
        } else {
            $('#ai-worker-panel').slideDown();
        }
    }

    _setup_markup() {

        this.workersContainer = $('#ai-worker-entries');
        this.tasksContainer = $('#ai-tasks-entries');

        // mini-panel global progress indicator for tasks
        this.prInd_tasks = $('<div class="minipanel-progress" style="display:none"></div>');
        this.miniStatus_tasks = $('<span style="margin-right:10px">Tasks:</span>');
        this.prInd_tasks.append(this.miniStatus_tasks);
        var pbarWrapper = $('<div class="progressbar" id="minipanel-progressbar"></div>');
        this.progressBar_tasks = $('<div class="progressbar-filler progressbar-active"></div>');
        pbarWrapper.append(this.progressBar_tasks);
        this.prInd_tasks.append(pbarWrapper);
        $('#ai-minipanel-status').append(this.prInd_tasks);

        // the same for annotations
        this.prInd_anno = $('<div class="minipanel-progress" style="display:none"></div>');
        this.miniStatus_anno = $('<span style="margin-right:10px">Annotations:</span>');
        this.prInd_anno.append(this.miniStatus_anno);
        var pbarWrapper = $('<div class="progressbar" id="minipanel-progressbar"></div>');
        this.progressBar_anno = $('<div class="progressbar-filler progressbar-active"></div>');
        pbarWrapper.append(this.progressBar_anno);
        this.prInd_anno.append(pbarWrapper);
        $('#ai-minipanel-status').append(this.prInd_anno);


        // make panel collapsible
        var self = this;
        $('#ai-worker-minipanel').click(function() {
            self.__toggle_panel();
        });
        $('#ai-worker-panel-header').click(function() {
            self.__toggle_panel();
        });

        // collapsible category headers
        $('#ai-worker-header').click(function() {
            if($(this).hasClass('expanded')) {
                $('#ai-worker-entries').slideUp();
            } else {
                $('#ai-worker-entries').slideDown();
            }
            $(this).toggleClass('expanded');
        });
        $('#ai-tasks-header').click(function() {
            if($(this).hasClass('expanded')) {
                $('#ai-tasks-entries').slideUp();
            } else {
                $('#ai-tasks-entries').slideDown();
            }
            $(this).toggleClass('expanded');
        });


        // manual control (if admin)
        if(window.isAdmin) {        // if(window.getCookie('isAdmin') === 'y') {
            $('#ai-manual-controls').append(
                $(`<table style="width:100%">
                    <tr>
                        <td><input type="checkbox" id="check-do-train" /></td><td>train</td>
                        <td># Images: <input type="number" id="box-num-images-train" class="number-box" min="0" max="8192" value="256" /></td>
                    </tr>
                    <tr>
                        <td></td><td></td><td>min # annotations/image: <input type="number" id="box-num-annotations-train" class="number-box" min="0" value="0" /></td>
                    </tr>
                    <tr>
                        <td><input type="checkbox" id="check-do-inference" /></td><td>inference</td>
                        <td># Images: <input type="number" id="box-num-images-inference" class="number-box" min="0" max="32768" value="1024" /></td>
                    </tr>
                </table>
                <button class="btn btn-primary btn-sm" id="launch-job-button" style="float:right;margin-top:10px;">Launch</button>`)
            );
            // disable class hotkey switching on text field focus
            $('#ai-manual-controls *').on({
                focusin: function() {
                    window.shortcutsDisabled = true;
                },
                focusout: function() {
                    window.shortcutsDisabled = false;
                }
            });
            $('#launch-job-button').click(function() {
                self.manualTrigger();
            })
        }
    }


    _query_worker_status() {
        var self = this;
        $.ajax({
            url: window.aiControllerURI + 'status?workers=true',
            type: 'GET',
            success: function(data) {
                // parse workers
                var workers = data['status']['workers'];
                
                for(var key in workers) {
                    if(!self.workers.hasOwnProperty(key)) {
                        var worker = new AIWorker(Object.keys(self.workers).length+1, key, workers[key]);
                        self.workers[key] = worker;
                        self.workersContainer.prepend(worker.markup);

                    } else {
                        // update worker
                        self.workers[key].update_state(workers[key]);
                    }
                }
                setTimeout(function() { self._query_worker_status(); }, 10000);   //TODO: make parameter
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // ask user to provide password again
                    window.verifyLogin((self._query_worker_status).bind(self));
                } else {
                    setTimeout(function() { self._query_worker_status(); }, 60000);   //TODO: make parameter
                }
            }
        });
    }


    _query_task_status() {
        var self = this;
        $.ajax({
            url: window.aiControllerURI + 'status?project=true&tasks=true',
            type: 'GET',
            success: function(data) {

                // check first if wait for sufficient number of annotations is ongoing
                if(data['status'].hasOwnProperty('project') && data['status']['project'].hasOwnProperty('num_annotated')) {
                    var numRequired_project = parseInt(data['status']['project']['num_next_training']);
                    var numDone_project = Math.min(numRequired_project, parseInt(data['status']['project']['num_annotated']));
                    var msg = numDone_project + '/' + numRequired_project + ' images until next training';
                    $(self.miniStatus_anno).html(msg);

                    // update progress bar for annotations
                    $(self.prInd_anno).show();
                    $(self.progressBar_anno).show();
                    $(self.miniStatus_anno).show();

                    var newWidthPerc = 100*(numDone_project / numRequired_project);
                    self.progressBar_anno.animate({
                        'width': newWidthPerc + '%'
                    }, 1000);

                } else {
                    // no project data; hide progress info
                    $(self.prInd_anno).hide();
                    $(self.progressBar_anno).hide();
                    $(self.miniStatus_anno).hide();
                }

                // global task status
                var numTasksInProgress = 0;
                var numTotal = 0;
                var numDone = 0;

                // parse tasks
                var tasks = data['status']['tasks'];
                
                for(var key in tasks) {
                    if(!self.tasks.hasOwnProperty(key)) {
                        var task = new AIWorkerJob(Object.keys(self.tasks).length+1, key, tasks[key]);
                        self.tasks[key] = task;
                        self.tasksContainer.prepend(task.markup);

                    } else {
                        // update task
                        self.tasks[key].update_state(tasks[key]);
                    }

                    // parse task progress
                    if(!(tasks[key].status === 'SUCCESS' || tasks[key].status === 'FAILURE')) {
                        numTasksInProgress += 1;
                        if(tasks[key].hasOwnProperty('meta') && tasks[key]['meta'] != null) {
                            if(tasks[key]['meta'].hasOwnProperty('total')) {
                                numTotal += tasks[key]['meta']['total'];
                            }
                            if(tasks[key]['meta'].hasOwnProperty('done')) {
                                numDone += tasks[key]['meta']['done'];
                            }
                        }
                    }
                }

                // update global progress bar and status message
                if(numTasksInProgress > 0) {
                    $(self.prInd_tasks).show();
                    $(self.progressBar_tasks).show();
                    $(self.miniStatus_tasks).show();
                    var msg = numTasksInProgress;
                    msg += (numTasksInProgress == 1 ? ' task' : ' tasks');
                    
                    if(numTotal > 0) {
                        var newWidthPerc = 100*(numDone / numTotal);
                        self.progressBar_tasks.animate({
                            'width': newWidthPerc + '%'
                        }, 1000);
                        msg += ' (' + Math.round(newWidthPerc) + '%)';
                    } else {
                        if(numTasksInProgress > 0) {
                            // tasks going on, but progress unknown, set indeterminate
                            self.progressBar_tasks.css('width', '100%');
                        } else {
                            // no task going on; hide progress bar
                            $(self.prInd_tasks).hide();
                            $(self.progressBar_tasks).hide();
                            $(self.miniStatus_tasks).hide();
                        }
                    }
                    $(self.miniStatus_tasks).html(msg);
                } else {
                    // no tasks; hide progress bar and status message
                    $(self.prInd_tasks).hide();
                    $(self.progressBar_tasks).hide();
                    $(self.miniStatus_tasks).hide();
                }

                var timeoutVal = 10000;
                if(numTasksInProgress > 0) {
                    // increase polling frequency
                    timeoutVal = 1000;
                }

                setTimeout(function() { self._query_task_status(); }, timeoutVal);
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // ask user to provide password again
                    window.verifyLogin((self._query_task_status).bind(self));
                } else {
                    setTimeout(function() { self._query_task_status(); }, 20000);   //TODO: make parameter
                }
            }
        });
    }

    
    startInference() {
        var self = this;
        $.ajax({
            url: 'startInference',
            method: 'POST',
            success: function(data) {
                // console.log(data);
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // ask user to provide password again
                    window.verifyLogin((self.startInference).bind(self));
                }
            }
        })
    }

    startTraining() {
        var self = this;
        $.ajax({
            url: 'startTraining',
            method: 'POST',
            success: function(data) {
                // console.log(data);
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // ask user to provide password again
                    window.verifyLogin((self.startTraining).bind(self));
                }
            }
        })
    }

    manualTrigger() {
        /*
            Gets the data entered through the interface and
            manually starts a training, inference, or combined
            cycle.
        */
        var self = this;
        var doTrain = $('#check-do-train').is(":checked");
        var numAnnoPerImg = $('#box-num-annotations-train').val();
        var numImg_train = $('#box-num-images-train').val();
        var doInference = $('#check-do-inference').is(":checked");
        var numImg_inference = $('#box-num-images-inference').val();

        //TODO: check if training process running and disable checkbox if so
        //TODO: default box values
        //TODO: also check for maximum number of (concurrent) processes

        if(!(doTrain || doInference)) return;
        var data = {
            'train': doTrain,
            'minNumAnnoPerImage': numAnnoPerImg,
            'maxNum_train': numImg_train,
            'inference': doInference,
            'maxNum_inference': numImg_inference
        };
        $.ajax({
            url: 'start',
            method: 'POST',
            data: JSON.stringify(data),
            contentType: 'application/json; charset=utf-8',
            dataType: 'json',
            success: function(response) {
                // console.log(response);  //TODO
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // ask user to provide password again
                    window.verifyLogin((self.manualTrigger).bind(self));
                }
            }
        })
    }
}