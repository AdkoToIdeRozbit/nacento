from django.shortcuts import render
from base.funcs import *
from rest_framework.response import Response
from rest_framework.decorators import api_view
from base64 import b64decode
from pdf2image import convert_from_bytes
import cv2, numpy

routes = [
        {
            "Endpoint" : "/aspdf/",
            "method" : "POST"
        }
    ]

@api_view(['GET', 'POST'])
def home(request, ):
    return Response(routes)

@api_view(['GET', 'POST'])
def get_pdf(request):
    print(request.method)
    if request.method == 'POST':

        data = request.data.replace('data:application/pdf;base64,', '')
        pdf = b64decode(data, validate=True)
        if pdf[0:4] != b'%PDF':
            raise ValueError('Missing the PDF file signature')

        image = numpy.array(convert_from_bytes(pdf)[0]) #konverzia len prvej strany pdf


       


    return Response(routes)



'''
def home(request):
    #img = cv2.imread('base/static/base/imgs/34.tiff', 0)
#img = cv2.imread('/home/adik/Desktop/TIFF DATA/ZAKLADY TIFF/60.tiff', 0) #56 is problem
    

    # stavba = get_stavba(img)
    # seg,img = segment_zaklady(stavba)

#seg, img = segment_zaklady(img)

    # img = cv2.imread('base/static/base/imgs/3.tiff')
    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #stavba, original = get_stavba(img)
    #stavba =cv2.cvtColor(stavba,cv2.COLOR_GRAY2RGB)
    #seg,img = segment_zaklady(stavba)


    #context = {'img' : img_to_html(seg)}
    #context = {'img' : img_to_html(seg), "original" : img_to_html(img)}
    #return render(request, 'base/home.html', context)
    return render(request, 'base/home.html')
'''