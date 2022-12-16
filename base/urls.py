from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/aspdf/', views.get_pdf, name='get_pdf'),
]
