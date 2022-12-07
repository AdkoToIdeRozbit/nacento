from django.shortcuts import render
from base.funcs import *


def home(request):
    #img = cv2.imread('base/static/base/imgs/5.tiff')
    img = cv2.imread('/home/adik/Desktop/TIFF_DATA/train/57.tiff') #56 is problem
    original = cv2.resize(img, (IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    stavba = get_stavba(img)
    
    stavba =cv2.cvtColor(stavba,cv2.COLOR_GRAY2RGB)
    seg = segment_zaklady(stavba)


    context = {'img' : img_to_html(seg), "original" : img_to_html(original)}
    return render(request, 'base/home.html', context)
