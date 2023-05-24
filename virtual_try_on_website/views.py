import base64
import time
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageUploadForm
from .Utils import readb64
from .PoseParser import pose_parse


def index(request):
    person_image_uri = None
    cloth_image_uri = None
    keypoints_dict = None

    form = ImageUploadForm()

    # pass the form, image URI, and predicted label to the template to be rendered
    context = {
        'form': form
    }
    return render(request, '.\\index.html', context)


def postImages(request):
    # request should be ajax and method should be POST.
    if request.method == "POST":
        # get the form data
        form = ImageUploadForm(request.POST, request.FILES)
        # for field in form:
        #     print("Field Error:", field.name, field.errors)
        # save the data and after fetch the object in instance
        if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and form.is_valid():
            image = form.cleaned_data['image_person']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            person_image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
            person_image_bytes = readb64(person_image_uri)
            keypoints_dict = pose_parse(person_image_bytes)

            return JsonResponse({"result": keypoints_dict}, status=200)
        else:
            # some form errors occured.
            return JsonResponse({"error": "xfdcgv"}, status=400)

    # some error occured
    return JsonResponse({"error": "wtf"}, status=400)


def postPose(request):
    print("pose")
    if request.method == "POST" and request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
        time.sleep(3)
        return JsonResponse({"result": "SUCCESS"}, status=200)
    else:
        return JsonResponse({"error": "error"}, status=400)
