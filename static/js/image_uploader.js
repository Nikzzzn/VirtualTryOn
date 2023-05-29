function readURL(input) {
    input_node = $(`#${input.id}`);
    file_upload_content_node = input_node.next();
    file_upload_image_node = file_upload_content_node.children('img');
    files_node = input_node.parent()
    input_node.removeClass('image-dropping');
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function(e) {
            input_node.hide();
            file_upload_image_node.attr('src', e.target.result);
            file_upload_content_node.show();
            files_node.addClass('no-pseudo');
        };
        reader.readAsDataURL(input.files[0]);
    } else {
        removeUpload(input.id);
    }
}

function removeUpload(input_id) {
    input_node = $(`#${input_id}`);
    file_upload_content_node = input_node.next();
    files_node = input_node.parent()
    file_upload_content_node.hide();
    input_node[0].value = null;
    input_node.show();
    console.log(input_node);
    files_node.removeClass('no-pseudo');
}
function dragEnter(element) {
    $(`#${element.id}`).addClass('image-dropping');
}
function dragLeave(element) {
    $(`#${element.id}`).removeClass('image-dropping');
}