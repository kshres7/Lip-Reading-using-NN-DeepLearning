/*
Calculate a standart deviation of array of values.
https://derickbailey.com/2014/09/21calculating-standard-deviation-with-array-map-and-array-reduce-in-javascript/ */
function standardDeviation(values){
    var avg = average(values);

    var squareDiffs = values.map(function(value){
        var diff = value - avg;
        var sqrDiff = diff * diff;
        return sqrDiff;
    });

    var avgSquareDiff = average(squareDiffs);
    var stdDev = Math.sqrt(avgSquareDiff);
    return stdDev;
}

function average(data){
    var sum = data.reduce(function(sum, value){
        return sum + value;
    }, 0);
    var avg = sum / data.length;
    return avg;
}

// Flatten array dimensions into 1D
function flatten(arr) {
    return arr.reduce(function (flat, toFlatten) {
        return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
    }, []);
}

// Credits to K3N - https://stackoverflow.com/a/19773537
function FpsCtrl(fps, callback) {

    var	delay = 1000 / fps,
        time = null,
        frame = -1,
        tref;

    function loop(timestamp) {
        if (time === null) time = timestamp;
        var seg = Math.floor((timestamp - time) / delay);
        if (seg > frame) {
            frame = seg;
            callback({
                time: timestamp,
                frame: frame
            })
        }
        tref = requestAnimationFrame(loop)
    }
    this.isPlaying = false;

    this.frameRate = function(newfps) {
        if (!arguments.length) return fps;
        fps = newfps;
        delay = 1000 / fps;
        frame = -1;
        time = null;
    };
    this.start = function() {
        if (!this.isPlaying) {
            this.isPlaying = true;
            tref = requestAnimationFrame(loop);
        }
    };
    this.pause = function() {
        if (this.isPlaying) {
            cancelAnimationFrame(tref);
            this.isPlaying = false;
            time = null;
            frame = -1;
        }
    };
}

// Calculate stats for Webcam Video FPS
function calculateStats() {

    var decodedFrames = 0,
            droppedFrames = 0,
            startTime = new Date().getTime(),
            initialTime = new Date().getTime();

    window.setInterval(function(){

        //see if webkit stats are available; exit if they aren't
        if (!vid.webkitDecodedFrameCount){
            //console.log("Video FPS calcs not supported");
            //console.log(vid.mozFrameDelay );
            return;
        }
        //get the stats
        else {
            var currentTime = new Date().getTime();
            var deltaTime = (currentTime - startTime) / 1000;
            var totalTime = (currentTime - initialTime) / 1000;
            startTime = currentTime;

            // Calculate decoded frames per sec.
            var currentDecodedFPS  = (vid.webkitDecodedFrameCount - decodedFrames) / deltaTime;
            var decodedFPSavg = vid.webkitDecodedFrameCount / totalTime;
            decodedFrames = vid.webkitDecodedFrameCount;

            // Calculate dropped frames per sec.
            var currentDroppedFPS = (vid.webkitDroppedFrameCount - droppedFrames) / deltaTime;
            var droppedFPSavg = vid.webkitDroppedFrameCount / totalTime;
            droppedFrames = vid.webkitDroppedFrameCount;

            //write the results to a table
            // $("#stats")[0].innerHTML =
            //     "<table><tr><th>Type</th><th>Total</th><th>Avg</th><th>Current</th></tr>" +
            //     "<tr><td>Decoded</td><td>" + decodedFrames + "</td><td>" + decodedFPSavg.toFixed() + "</td><td>" + currentDecodedFPS.toFixed()+ "</td></tr>" +
            //     "<tr><td>Dropped</td><td>" + droppedFrames + "</td><td>" + droppedFPSavg.toFixed() + "</td><td>" + currentDroppedFPS.toFixed() + "</td></tr>" +
            //     "<tr><td>All</td><td>" + (decodedFrames + droppedFrames) + "</td><td>" + (decodedFPSavg + droppedFPSavg).toFixed() + "</td><td>" + (currentDecodedFPS + currentDroppedFPS).toFixed() + "</td></tr>" +
            //     "<tr><td colspan='4'>Camera resolution: " + vid.videoWidth + " x " + vid.videoHeight + "</td><td></table>";
            if (currentDecodedFPS < 23) {
                $("#fps")[0].innerHTML = "WARNING: Low camera FPS, prediction may not work<br>";
                $("#fps")[0].innerHTML += currentDecodedFPS.toFixed() + " FPS";
                $("#fps").css('color', 'red');
            } else {
                $("#fps")[0].innerHTML = currentDecodedFPS.toFixed() + " FPS";
                $("#fps").css('color', 'green');
            }
        }
    }, 1000);
}

/*********** Setup of video/webcam and checking for webGL support *********/
function enablestart() {
    var startbutton = document.getElementById('startbutton');
    startbutton.value = "start";
    startbutton.disabled = null;
}

var insertAltVideo = function(video) {
    // insert alternate video if getUserMedia not available
    if (supports_video()) {
        if (supports_webm_video()) {
            video.src = "./media/cap12_edit.webm";
        } else if (supports_h264_baseline_video()) {
            video.src = "./media/cap12_edit.mp4";
        } else {
            return false;
        }
        return true;
    } else return false;
}

function adjustVideoProportions() {
    // resize overlay and video if proportions of video are not 4:3
    // keep same height, just change width
    var proportion = vid.videoWidth/vid.videoHeight;
    vid_width = Math.round(vid_height * proportion);
    vid.width = vid_width;
    overlay.width = vid_width;
}

function gumSuccess( stream ) {
    // add camera stream if getUserMedia succeeded
    if ("srcObject" in vid) {
        vid.srcObject = stream;
    } else {
        vid.src = (window.URL && window.URL.createObjectURL(stream));
    }
    vid.onloadedmetadata = function() {
        adjustVideoProportions();
        vid.play();
    }
    vid.onresize = function() {
        adjustVideoProportions();
        if (trackingStarted) {
            ctrack.stop();
            ctrack.reset();
            ctrack.start(vid);
        }
    }
}

function gumFail() {
    // fall back to video if getUserMedia failed
    insertAltVideo(vid);
    document.getElementById('gum').className = "hide";
    document.getElementById('nogum').className = "nohide";
    alert("There was some problem trying to fetch video from your webcam, using a fallback video instead.");
}

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
window.URL = window.URL || window.webkitURL || window.msURL || window.mozURL;

// set up video
if (navigator.mediaDevices) {
    navigator.mediaDevices.getUserMedia({video : true}).then(gumSuccess).catch(gumFail);
} else if (navigator.getUserMedia) {
    navigator.getUserMedia({video : true}, gumSuccess, gumFail);
} else {
    insertAltVideo(vid);
    document.getElementById('gum').className = "hide";
    document.getElementById('nogum').className = "nohide";
    alert("Your browser does not seem to support getUserMedia, using a fallback video instead.");
}

/*********** Code for stats **********/
stats = new Stats();
stats.domElement.style.position = 'absolute';
stats.domElement.style.top = '0px';
document.getElementById('container').appendChild( stats.domElement );

// update stats on every iteration
document.addEventListener('clmtrackrIteration', function(event) {
    stats.update();
}, false);
