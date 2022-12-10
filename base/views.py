from django.shortcuts import render
from base.funcs import *


def home(request):
    img = cv2.imread('base/static/base/imgs/5.tiff')
    #img = cv2.imread('/home/adik/Desktop/TIFF DATA/ZAKLADY TIFF/5.tiff') #56 is problem
    
    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    # stavba, original = get_stavba(img)
    # stavba =cv2.cvtColor(stavba,cv2.COLOR_GRAY2RGB)
    # seg,img = segment_zaklady(stavba)


    seg, img = segment_zaklady(img)


    #context = {'img' : img_to_html(seg)}
    context = {'img' : img_to_html(seg), "original" : img_to_html(img)}
    return render(request, 'base/home.html', context)
