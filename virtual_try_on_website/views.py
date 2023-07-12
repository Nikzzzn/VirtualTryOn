import json

import skimage.io
from django.http import JsonResponse
from django.shortcuts import render
from .forms import ImageUploadForm
from .Utils import *
from .processors.lip_jppnet import LIP_JPPNet
from .processors.u2netp import U2NETP_Processor
from .processors.openpose import OpenPose_Processor
from .processors.cp_vton import CP_VTON_Processor
from .processors.viton_hd import VITON_HD_Processor
from .processors.hr_viton import HR_VITON_Processor

lip_jppnet_model = LIP_JPPNet()
u2netp_model = U2NETP_Processor()
openpose_model = OpenPose_Processor()
cp_vton_model = CP_VTON_Processor()
viton_hd_model = VITON_HD_Processor()
hr_viton_model = HR_VITON_Processor()


def index(request):
    form = ImageUploadForm()

    context = {
        'form': form
    }
    return render(request, 'index.html', context)


def post_calculate_person_pose(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        # if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and form.is_valid():
        if form.is_valid():
            person_image = read_uploaded_image(form.cleaned_data['image_person'])
            if form.cleaned_data['method'] == 'CP-VTON+':
                person_image = crop_image(person_image, 256, 192)
                keypoints_dict = openpose_model.pose_parse(person_image, n_points=18)
            else:
                person_image = crop_image(person_image, 1024, 768)
                keypoints_dict = openpose_model.pose_parse(person_image)

            return JsonResponse({"result": keypoints_dict}, status=200)
        else:
            return JsonResponse({"error": "xfdcgv"}, status=400)

    return JsonResponse({"error": "wtf"}, status=400)


def post_calculate_segmentation(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        # if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and form.is_valid():
        if form.is_valid():
            person_image = read_uploaded_image(form.cleaned_data['image_person'], color='bgr')
            if form.cleaned_data['method'] == 'CP-VTON+':
                person_image = crop_image(person_image, 256, 192)
            else:
                person_image = crop_image(person_image, 1024, 768)
            image_base64 = lip_jppnet_model.parse_person(person_image)
            image_uri = 'data:%s;base64,%s' % ('image/png', image_base64)

            return JsonResponse({"result": image_uri}, status=200)
        else:
            return JsonResponse({"error": "error"}, status=400)
    else:
        return JsonResponse({"error": "error"}, status=400)


def post_calculate_cloth_mask(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        # if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and form.is_valid():
        if form.is_valid():
            cloth_image = read_uploaded_image(form.cleaned_data['image_cloth'])
            if form.cleaned_data['method'] == 'CP-VTON+':
                cloth_image = crop_image(cloth_image, 256, 192)
            else:
                cloth_image = crop_image(cloth_image, 1024, 768)
            mask_base64 = u2netp_model.calculate_mask(cloth_image)
            image_uri = 'data:%s;base64,%s' % ('image/jpg', mask_base64)

            return JsonResponse({"result": image_uri}, status=200)
        else:
            return JsonResponse({"error": "error"}, status=400)
    else:
        return JsonResponse({"error": "error"}, status=400)


def post_generate_result(request):
    required_datakeys = ["pose_keypoints", "person_segmentation", "cloth_mask"]

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        # if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and form.is_valid():
        if form.is_valid():
            if all(map(lambda x: x in request.POST, required_datakeys)):
                for key in required_datakeys:
                    form.cleaned_data[key] = request.POST[key]
            else:
                return JsonResponse({"error": "error"}, status=400)

            person_image = read_uploaded_image(form.cleaned_data['image_person'])
            cloth_image = read_uploaded_image(form.cleaned_data['image_cloth'])

            pose_keypoints = json.loads(form.cleaned_data["pose_keypoints"])

            person_segmentation = read_base64(form.cleaned_data["person_segmentation"], mode='pil')
            cloth_mask = read_base64(form.cleaned_data["cloth_mask"], grayscale=True)

            if form.cleaned_data["method"] == "CP-VTON+":
                model = cp_vton_model
                person_image = crop_image(person_image, 256, 192)
                cloth_image = crop_image(cloth_image, 256, 192)
            elif form.cleaned_data["method"] == "VITON-HD":
                model = viton_hd_model
                person_image = crop_image(person_image, 1024, 768)
                cloth_image = crop_image(cloth_image, 1024, 768)
            elif form.cleaned_data["method"] == "HR-VITON":
                model = hr_viton_model
                person_image = crop_image(person_image, 1024, 768)
                cloth_image = crop_image(cloth_image, 1024, 768)
            else:
                return JsonResponse({"error": "error"}, status=400)

            image_base64 = model.generate_result(person_image, cloth_image, pose_keypoints, person_segmentation, cloth_mask)
            image_uri = 'data:%s;base64,%s' % ('image/jpg', image_base64)

            return JsonResponse({"result": image_uri}, status=200)
        else:
            return JsonResponse({"error": "error"}, status=400)
    else:
        return JsonResponse({"error": "error"}, status=400)


def crop_image(in_image, out_height, out_width):
    in_height, in_width, _ = in_image.shape

    if in_height == out_height and in_width == out_width:
        return in_image

    in_ratio = in_height / in_width
    out_ratio = out_height / out_width

    if np.isclose(in_ratio, out_ratio):
        return cv2.resize(in_image, (out_width, out_height))
    else:
        # keypoints = openpose_model.pose_parse(in_image, n_points=25)
        #
        # rect_x0, rect_y0, rect_a, rect_b = openpose_model.get_rectangle(keypoints=keypoints, threshold=0.1)
        # rect_x1 = rect_x0 + rect_a
        # rect_y1 = rect_y0 + rect_b
        # rect_height = abs(rect_y0 - rect_y1)
        # rect_width = abs(rect_x0 - rect_x1)
        # rect_ratio = rect_height / rect_width

        # if in_height > rect_height > out_height:
        #     if out_ratio <= rect_ratio:
        #         temp_y0, temp_y1 = rect_y0, rect_y1
        #         temp_height = abs(temp_y0 - temp_y1)
        #         temp_width = temp_height / out_ratio
        #
        #         temp_x0 = (rect_x0 + rect_x1) / 2 - temp_width / 2
        #         temp_x1 = (rect_x0 + rect_x1) / 2 + temp_width / 2
        #     else:
        #         temp_x0, temp_x1 = rect_x0, rect_x1
        #         temp_width = abs(temp_x0 - temp_x1)
        #         temp_height = temp_width * out_ratio
        #
        #         temp_y0 = (rect_y0 + rect_y1) / 2 - temp_width / 2
        #         temp_y1 = (rect_y0 + rect_y1) / 2 + temp_width / 2
        #     image_cropped = in_image[int(temp_y0):int(temp_y1), int(temp_x0):int(temp_x1)]
        #     skimage.io.imshow(image_cropped)
        #     return cv2.resize(image_cropped, (out_width, out_height))
        return cv2.resize(in_image, (out_width, out_height))
