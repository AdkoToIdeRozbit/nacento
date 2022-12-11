from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView


urlpatterns = [
    path('admin/', admin.site.urls),
    #path('', include('base.urls')),
    path('', TemplateView.as_view(template_name='index.html'))

]
