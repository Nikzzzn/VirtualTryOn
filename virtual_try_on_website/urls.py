"""
URL configuration for virtual_try_on_website project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('post/ajax/calculate_pose', views.post_calculate_person_pose, name="calculate_pose"),
    path('post/ajax/calculate_segmentation', views.post_calculate_segmentation, name="calculate_segmentation"),
    path('post/ajax/calculate_mask', views.post_calculate_cloth_mask, name="calculate_mask"),
    path('post/ajax/generate_result', views.post_generate_result, name="generate_result"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
