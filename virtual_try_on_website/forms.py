from django import forms


class ImageUploadForm(forms.Form):
    CHOICES = (('CP-VTON+', 'CP-VTON+'), ('VITON-HD', 'VITON-HD'), ('HR-VITON', 'HR-VITON'))

    image_person = forms.ImageField(label="Person")
    image_cloth = forms.ImageField(label="Cloth")
    method = forms.ChoiceField(choices=CHOICES)

