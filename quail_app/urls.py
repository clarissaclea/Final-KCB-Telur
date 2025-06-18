from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dataset/', views.dataset_view, name='dataset'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('testing/', views.testing_view, name='testing'),
    path('klasifikasi/', views.klasifikasi_view, name='klasifikasi'),
]
