import base64
import json
from django.http import JsonResponse
from django.shortcuts import render
from .forms import ImageUploadForm
from .Utils import *
from .PoseParser import pose_parse
from virtual_try_on_website.processors.lip_jppnet import LIP_JPPNet
from virtual_try_on_website.processors.u2netp import U2NETP_Processor
from .processors.viton import ViTON_Processor

lip_jppnet_model = LIP_JPPNet()
u2netp_model = U2NETP_Processor()
viton_model = ViTON_Processor()


def index(request):
    form = ImageUploadForm()

    context = {
        'form': form
    }
    return render(request, '.\\index.html', context)


def post_calculate_person_pose(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and form.is_valid():
            # image = form.cleaned_data['image_person']
            # image_bytes = image.file.read()
            # encoded_img = base64.b64encode(image_bytes).decode('ascii')  # TODO: clear bs
            # person_image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
            # person_image_bytes = read_base64(person_image_uri)
            person_image = read_uploaded_image(form.cleaned_data['image_person'])
            keypoints_dict = pose_parse(person_image)

            return JsonResponse({"result": keypoints_dict}, status=200)
        else:
            return JsonResponse({"error": "xfdcgv"}, status=400)

    return JsonResponse({"error": "wtf"}, status=400)


def post_calculate_segmentation(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and form.is_valid():
            person_image = read_uploaded_image(form.cleaned_data['image_person'], color='bgr')
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
        if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and form.is_valid():
            cloth_image = read_uploaded_image(form.cleaned_data['image_cloth'])
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
        if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and form.is_valid():
            if all(map(lambda x: x in request.POST, required_datakeys)):
                for key in required_datakeys:
                    form.cleaned_data[key] = request.POST[key]
            else:
                return JsonResponse({"error": "error"}, status=400)

            person_image = read_uploaded_image(form.cleaned_data['image_person'])
            cloth_image = read_uploaded_image(form.cleaned_data['image_cloth'])

            pose_keypoints = json.loads(form.cleaned_data["pose_keypoints"])
            pose_keypoints = pose_keypoints['people'][0]['pose_keypoints']

            person_segmentation = read_base64(form.cleaned_data["person_segmentation"], mode='pil')
            cloth_mask = read_base64(form.cleaned_data["cloth_mask"], grayscale=True)

            image_base64 = viton_model.generate_result(person_image, cloth_image, pose_keypoints, person_segmentation, cloth_mask)
            image_uri = 'data:%s;base64,%s' % ('image/jpg', image_base64)

            return JsonResponse({"result": image_uri}, status=200)
        else:
            return JsonResponse({"error": "error"}, status=400)
    else:
        return JsonResponse({"error": "error"}, status=400)