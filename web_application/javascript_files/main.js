// Dimensions of 3D lip matrix
var IMG_WIDTH = 32;
var IMG_HEIGHT = 24;
var VID_DEPTH = 28;


var MODEL = './model_D.json';
var MODEL_WEIGHTS = './model_D_weight.buf';
var MODEL_METADATA = 'model_D_metadata.json';

var words = ["ABOUT", "ACCESS", "ALLOW", "BANKS", "BLACK", "CALLED", "CONCERNS",
            "CRISIS", "DEGREES", "DIFFERENT", "DOING", "EDITOR", "ELECTION",
            "EVERY", "FOCUS", "GROUP", "HUMAN", "IMPACT", "JUSTICE"];

var recordingState = false;
var predicting = false;

// variables for measuring time
var t0 = 0;
var t1 = 0;

var myChart;
let model;

var mouth_brightness_array = new Array(10);
for (var i = 0; i < 10; i++) {
    mouth_brightness_array[i] = 1;
}
mouth_brightness_mean = 0;

var mouth_openness_array = new Array(VID_DEPTH);
for (var i = 0; i < VID_DEPTH; i++) {
    mouth_openness_array[i] = 1;
}

var config = {
    type: 'line',
    data: {
        labels: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
        datasets: [
        {
            label: 'mouth brightness std. deviation',
            data: [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
            borderColor: [
                'rgba(0,255,255,1)'
            ],
            borderWidth: 1
        }],
    },
    options: {
        scales: {
            yAxes: [{
                ticks: {
                    min: 0,
                    max: 30
                }
            }]
        },
        elements: {
            line: {
                tension: 0, // disables bezier curves
            }
        },
        animation: {
            duration: 0, // general animation time
        },
        events: [],
        responsiveAnimationDuration: 0, // animation duration after a resize
    }
}

var ctx = document.getElementById("mouthOpenCanvas").getContext('2d');
window.myLine = new Chart(ctx, config);
var mouthOpenChart = new Chart(ctx, config);

function queue(arr, item) {
  return arr.push(item), arr.shift();
}

function updateMouthChart(item, type) {
	config.data.datasets.forEach(function(dataset) {
        if (dataset.label == type){
            queue(dataset.data, item);
        }
	});
	window.myLine.update();
}


// Detect keyspace and go to recordingState
document.body.onkeydown  = function(e){
    if(e.keyCode == 32){
        recordingState = !recordingState;
        e.preventDefault();
    }
}

function updateChart(chart, newData) {
    chart.data.datasets[0].data = newData;
    chart.update();
}

var start_open = 0;
var end_open = 0;

window.onload = async function(){

    $("#toggleDebug").click(function() {
        $('.title').toggle();
        $('#mouthOpenCanvas').toggle();
    });

    var ctx = document.getElementById("myChart").getContext('2d');
    myChart = new Chart(ctx, {
        type: 'horizontalBar',
        data: {
            labels: words,
            datasets: [{
                data: Array.apply(null, Array(words.length)).map(Number.prototype.valueOf,0),
                label: 'Word Prediction Scale',
                backgroundColor: 'green',
            }]
        },
        options: {
            legend: {
                position: 'bottom',
            },
            scales: {
                xAxes: [{
                    display: true,
                    ticks: {
                        beginAtZero: true,
                        steps: 10,
                        stepValue: 10,
                        max: 100
                    }
                }],
            },
        }
    });

    // show help screen after 3 second, if first time web visit
    setTimeout(function() { showHelpFirstTime(); }, 3000);
    // start face tracking with 2 second delay
    setTimeout(function() { startVideo(); }, 2000);
    calculateStats();

    model = new KerasJS.Model({
        filepaths: {
            model: MODEL,
            weights: MODEL_WEIGHTS,
            metadata: MODEL_METADATA
        },
        gpu: true
    });
    await model.ready();
}
async function predict(arr) {

    const inputData = {
        input: new Float32Array(arr)
    }
    console.log(inputData);
    let outputData = await model.predict(inputData);
    console.log(outputData);

    var outputDataArray = Array.prototype.slice.call(outputData["output"]);
    var outputDataArrayInt = [];
    for(var i = 0; i < outputDataArray.length; i++) {
        var whole =  parseFloat(outputDataArray[i]);
        outputDataArrayInt.push(Math.round(whole * 100));
    }
    updateChart(myChart, outputDataArrayInt);

    predicting = false;
}

var vid = document.getElementById('videoel');
var vid_width = vid.width;
var vid_height = vid.height;
var overlay = document.getElementById('overlay');
var overlayCC = overlay.getContext('2d');

var mouthCanvas = document.getElementById("mouthCanvas");
var mouthCtx = mouthCanvas.getContext("2d");

var debugCanvas = document.getElementById("debugCanvas");
var debugCtx = debugCanvas.getContext("2d");

// Init face tracker
var ctrack = new clm.tracker();
ctrack.init();
var trackingStarted = false;

// Start video and face tracking
function startVideo() {
    // start video
    vid.play();
    // start tracking
    ctrack.start(vid);
    trackingStarted = true;
    // start loop to draw face
    fps.start();
    $(".spinner").hide();
}

// Init sample 3D array (VID_DEPTH x IMG_WIDTH x IMG_HEIGHT) and populate with 0
var lip_sample = new Array(VID_DEPTH);
for (var i = 0; i < VID_DEPTH; i++) {
    lip_sample[i] = Array.apply(null, Array(IMG_WIDTH * IMG_HEIGHT)).map(function () { return 0; })
}

var currentFrame = 0;
var DEBUG_ROW = 5;
var DEBUG_COL = 5;

var sum = 0;
for (var i = 0; i < mouth_openness_array.length; i++) {
    sum += mouth_openness_array[i];
}

// cols x rows
function getCenterMouthAreaBrightness(frame) {
    var x_center = IMG_WIDTH/2;
    var y_center = IMG_HEIGHT/2;
    var sum = 0;
    var cnt = 0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -5; x <= 5; x++) {
            sum += frame[y_center - y][x_center - x];
            cnt++;
        }
    }
    return Math.floor(sum/cnt);
}

// Main - run loop at specified FPS (25)
var fps = new FpsCtrl(25, function(e) {

    overlayCC.clearRect(0, 0, vid_width, vid_height);

    if (recordingState) {
        overlayCC.strokeStyle = "#ff0000";
        overlayCC.lineWidth=4;
        overlayCC.strokeRect(2, 2,vid_width - 2, vid_height - 2);
        overlayCC.lineWidth=1;
    }

    if (ctrack.getCurrentPosition()) {
        ctrack.draw(overlay);

        var pos = ctrack.getCurrentPosition();

        var mouth_left_top_corner = [pos[44][0], pos[46][1]];
        var mouth_right_top_corner = [pos[50][0], pos[46][1]];
        var mouth_width = mouth_right_top_corner[0] - mouth_left_top_corner[0];
        var mouth_height = pos[53][1] - pos[47][1];

        overlayCC.rect(
            mouth_left_top_corner[0] - 30,
            mouth_left_top_corner[1] - 30,
            mouth_width + 2*30,
            mouth_height + 2*30
        );
        overlayCC.strokeStyle = '#ff0000';
        overlayCC.stroke();

        var vidRatioX = vid.width / vid.videoWidth;
        var vidRatioY = vid.height / vid.videoHeight;


        mouthCtx.drawImage(vid,
            (mouth_left_top_corner[0] - 15) / vidRatioX,
            (mouth_left_top_corner[1] - 15) / vidRatioY,
            (mouth_width + 30) / vidRatioX,
            (mouth_height + 30) / vidRatioY,
            0, 0, mouthCanvas.width, mouthCanvas.height
        );

        var lipFrame = [];
        var lipRow = [];

        var imgd = mouthCtx.getImageData(0, 0, mouthCanvas.width, mouthCanvas.height);
        var pix = imgd.data;

        var centerMouthBrightness = 0;
        for (var i = 0, n = pix.length; i < n; i += 4) {
            var grayscale = pix[i ] * .3 + pix[i+1] * .59 + pix[i+2] * .11;
            pix[i ] = grayscale; // red
            pix[i+1] = grayscale; // green
            pix[i+2] = grayscale; // blue
            // alpha

            lipRow.push(Math.round(grayscale));
            if (lipRow.length == IMG_WIDTH){
                lipFrame.push(lipRow);
                lipRow = [];
            }
        }
        lip_sample.shift();
        lip_sample.push(lipFrame);

        mouthCtx.putImageData(imgd, 0, 0);
        debugCtx.putImageData(imgd, 0, 0);


        // if not predicting previous sample and spacebar was hit

        if (recordingState && !predicting) {
            // detect mouth opening
            if (br_std > 1.5) {

                t0 = performance.now();
                // takes into account 4 lip frames before opening mouth
                currentFrame = 7;
                predicting = true;
                recordingState = false;
            }
        }
        if (currentFrame != 0) {
            if (currentFrame == VID_DEPTH - 1){
                // show processing animation overlay
                $(".spinner").show();
            }
            if (currentFrame == VID_DEPTH) {

                currentFrame = 0;
                var tmp_slice = flatten(lip_sample);

                t1 = performance.now();
                console.log("Obtaining sample took " + (t1 - t0) + " milliseconds.")

                $(".spinner").show();

                // normalize array
                normalized = tmp_slice.map(function(x) {
                    return (x - 128) / 128 ;
                });

                t0 = performance.now();

                predict(normalized);

                t1 = performance.now();
                console.log("Predtiction took " + (t1 - t0) + " milliseconds.")

                $(".spinner").hide();
            } else {
                currentFrame++;
            }
        }
    }
})
