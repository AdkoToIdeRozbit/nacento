from django.shortcuts import render
from base.funcs import *


def home(request):
    #img = cv2.imread('base/static/base/imgs/34.tiff', 0)
    img = cv2.imread('/home/adik/Desktop/TIFF DATA/ZAKLADY TIFF/60.tiff', 0) #56 is problem
    

    # stavba = get_stavba(img)
    # seg,img = segment_zaklady(stavba)

    seg, img = segment_zaklady(img)

    # img = cv2.imread('base/static/base/imgs/3.tiff')
    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #stavba, original = get_stavba(img)
    #stavba =cv2.cvtColor(stavba,cv2.COLOR_GRAY2RGB)
    #seg,img = segment_zaklady(stavba)




    #context = {'img' : img_to_html(seg)}
    context = {'img' : img_to_html(seg), "original" : img_to_html(img)}
    return render(request, 'base/home.html', context)
