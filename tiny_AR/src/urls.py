from django.urls import path, include
from .views import (
    VideoView,
    indexscreen
)


urlpatterns = [
    path('stream/', VideoView, name='video-stream'),
    path('', indexscreen),
]
