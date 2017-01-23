from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^helloworld$', views.index, name='index'),
    url(r'^getdata$', views.getdata, name='getdata'),
    url(r'^aapl$', views.aapl, name='aapl'),
    url(r'^ajax', views.ajax, name='ajax'),
    url(r'^oror', views.oror, name='oror'),
]
