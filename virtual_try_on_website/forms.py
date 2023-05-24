from django import forms


class ImageUploadForm(forms.Form):
    CHOICES = (('ViTON', 'ViTON'), ('CP-VTON+', 'CP-VTON+'), ('HR-VTON', 'HR-VTON'))

    image_person = forms.ImageField(label="Person")
    image_cloth = forms.ImageField(label="Cloth")
    method = forms.ChoiceField(choices=CHOICES)

