let spinner_states = {};

function init(){
    spinner_states = {
        "pose": "Calculating pose keypoints...",
        "segmentation": "Calculating segmentation regions...",
        "mask": "Calculating cloth mask..."
    };
    $("#result-image").hide();
}

function actualize_spinner_state(completed_state){
    delete spinner_states[completed_state];
    let spinner_keys = Object.keys(spinner_states);
    if(spinner_keys.length > 0){
        $("#spinner-text").html(spinner_states[spinner_keys[0]]);
    }
    else {
        $("#spinner-text").html("Generating the final image...");
    }
}

function calculate_pose(data) {
    return $.ajax({
        url         : 'post/ajax/calculate_pose',
        type        : 'POST',
        data        : data,
        cache       : false,
        contentType : false,
        processData : false,
        success : function(data) {
            console.log('calculate_pose success');
        },
        error: function(data) {
            console.log('calculate_pose failed');
            console.log(data);
        },
        complete: function(data) {
            actualize_spinner_state('pose');
        }
    });
}

function calculate_segmentation(data) {
    return $.ajax({
        url         : 'post/ajax/calculate_segmentation',
        type        : 'POST',
        data        : data,
        cache       : false,
        contentType : false,
        processData : false,
        success : function(data) {
            console.log('calculate_segmentation success');
        },
        error: function(data) {
            console.log('calculate_segmentation failed');
            console.log(data);
        },
        complete: function(data) {
            actualize_spinner_state('segmentation');
        }
    });
}

function calculate_mask(data) {
    return $.ajax({
        url         : 'post/ajax/calculate_mask',
        type        : 'POST',
        data        : data,
        cache       : false,
        contentType : false,
        processData : false,
        success : function(data) {
            console.log('calculate_mask success');
        },
        error: function(data) {
            console.log('calculate_mask failed');
            console.log(data);
        },
        complete: function(data) {
            actualize_spinner_state('mask');
        }
    });
}

$("#inputform").on('submit', function(e){
    e.preventDefault();
    init();
    let spinner_node = $("#spinner-div");
    let spinner_text_node = $("#spinner-text");
    let formdata = new FormData(this);
    spinner_node.show();
    spinner_text_node.html(spinner_states['pose']);
    $.when(calculate_pose(formdata), calculate_segmentation(formdata), calculate_mask(formdata)).done(
        function(pose_data, segmentation_data, mask_data){
            if(pose_data[1] === "success" && segmentation_data[1] === "success" && mask_data[1] === "success"){
                formdata.append("pose_keypoints", JSON.stringify(pose_data[0]["result"]));
                formdata.append("person_segmentation", JSON.stringify(segmentation_data[0]["result"]));
                formdata.append("cloth_mask", JSON.stringify(mask_data[0]["result"]));
                spinner_node.show();

                $.ajax({
                    url         : 'post/ajax/generate_result',
                    type        : 'POST',
                    data        : formdata,
                    cache       : false,
                    contentType : false,
                    processData : false,
                    success : function(data) {
                        $("#result-image").show();
                        spinner_node.hide();
                        console.log('generate_result success');
                        parser_img_node = $("#result-image");
                        parser_img_node.attr('src', data.result);
                    },
                    error: function(data) {
                        console.log('generate_result failed');
                        console.log(data);
                        $("#spinner-div").hide();
                    }
                });
            }
        }
    );
});