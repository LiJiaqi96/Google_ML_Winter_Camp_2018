from django.db import models

class Profile(models.Model):
    picture = models.ImageField(upload_to = 'test_pictures')

    class Meta:
        db_table = "profile"

    def __str__(self):
        return self.name


