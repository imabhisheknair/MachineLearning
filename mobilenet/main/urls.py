from . import views
from django.urls import path

urlpatterns = [
    path('', views.home),
    path('find', views.find),
]
