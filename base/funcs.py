import segmentation_models as sm
from segmentation_models.losses  import bce_jaccard_loss
from segmentation_models.metrics import IOUScore
import easyocr 
import tensorflow as tf
import numpy as np
import base64
import cv2


IMG_WIDTH = 1120 #divisible by 32
IMG_HEIGHT = 704
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

model = tf.keras.models.load_model(
'base/static/base/nns/unet.hdf5', 
    custom_objects={
        'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss,
        'iou_score': IOUScore(),
    }
)
reader = easyocr.Reader(['en'], gpu=True) #more than 10x faster on GPU, better thean pytesseract on numbers
    

def img_to_html(img):
    frame_buff = cv2.imencode('.png', img)[1]
    frame_b64 = base64.b64encode(frame_buff).decode()
    return frame_b64


def segment_zaklady(stavba):
    print(stavba.shape)
    DX = stavba.shape[0] / IMG_HEIGHT
    DY = stavba.shape[1] / IMG_WIDTH
    original_stavba = stavba.copy()

    stavba = cv2.resize(stavba, IMG_SIZE)
    stavba = cv2.cvtColor(stavba,cv2.COLOR_GRAY2RGB)
    stavba = np.expand_dims(stavba, 0)
    prediction = model.predict(stavba)
    prediction = prediction[0,:,:,0]

    zaklady = np.zeros((prediction.shape) , np.uint8)
    for x in range(prediction.shape[0]):
        for y in range(prediction.shape[1]):
            if prediction[x][y] > 0.5 : zaklady[x][y] = 255
            else: zaklady[x][y] = 0

    # coef_y = img.shape[0] / IMG_HEIGHT
    # coef_x = img.shape[1] / IMG_WIDTH

    contours = cv2.findContours(zaklady, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    approx_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 200 : continue

        # cnt[:, :, 0] = cnt[:, :, 0] * coef_x
        # cnt[:, :, 1] = cnt[:, :, 1] * coef_y

        #peri = cv2.arcLength(cnt, True)
        #approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
        approx_contours.append(cnt)

    new_prediction = np.zeros((zaklady.shape) , np.uint8)
    cv2.fillPoly(new_prediction, pts=approx_contours, color=255)

    contours = np.concatenate(approx_contours) #group all contours
    x, y, w, h = cv2.boundingRect(contours)
    x,y,w,h = int(round(x*DY)), int(round(y*DX)), int(round(w*DY)), int(round(h*DX))
    cut = zisti_mierku(original_stavba, x, y, w, h)

    new_prediction = cv2.resize(new_prediction, (original_stavba.shape[1], original_stavba.shape[0]), interpolation=cv2.INTER_NEAREST)
    # print(np.unique(new_prediction))

    # area = cv2.countNonZero(new_prediction)
    # print(area*81)

    return cut, new_prediction

def find_nearest_white(img, target):
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index][0]

def find_end_dim_x(img, direction, x, Y):
    if direction: add = 1
    else : add = -1

    pixel_count = 0
    while(pixel_count < 5):
        pixel_count = 0
        for y in range(Y, Y + 5):
            if img[y][x] == 255 : pixel_count += 1
        x+=add

    return x

def search_dimensions(img1):
    original = img1.copy()
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    result = reader.readtext(img1)

    img1 = cv2.threshold(img1,220,255,0)[1]
    img1 = cv2.bitwise_not(img1)
    
    koty = []
    for i, x in enumerate(result):
        if x[1].isdigit() and 100000 > int(x[1]) > 0: #dimension (0,100) in meters
            koty.append((int(x[1]), i))

    koty = sorted(koty, reverse=True)

    if len(koty) == 0 : return img1

    for i in range(1):
        length = int(result[koty[i][1]][1])

        lh, ph, pd, ld = result[koty[i][1]][0]
        lh = (int(lh[0]), int(lh[1]))
        pd = (int(pd[0]), int(pd[1]))

        for x in range(lh[1], pd[1]):
            for y in range(lh[0], pd[0]):
                img1[x][y] = 0

        mid = (int(lh[0] + ((pd[0] - lh[0])/2)) ,  int(lh[1] + ((pd[1] - lh[1])/2)) )

        z = find_nearest_white(img1, mid)
        while(img1[z[1]][z[0]] == 255) : z[1] += 1
        z[1] -= 1


        x1 = find_end_dim_x(img1, True, z[0], z[1])
        x2 = find_end_dim_x(img1, False, z[0], z[1])

        if z[0] - x2 > 5 * (x1 - z[0]) : x1 = find_end_dim_x(img1, True, x1 + 25, z[1]) #fix intersected dimension
        if x1 - z[0] > 5 * (z[0] - x2) : x2 = find_end_dim_x(img1, False, x2 - 25, z[1])

        mierka = length / (x1 - x2) 
        print(mierka)
       
        cv2.line(original, (x1, z[1]), (x2, z[1]), (0,255,0), 1)
        cv2.rectangle(original, lh, pd, (0, 255, 0), 1)

    return original

def zisti_mierku(img, x, y, w, h):
    f = 15
    img1 = img[0:y, x-f:x+w+f]
    img2 = img[y+h:img.shape[0]-1, x-f:x+w+f]
    img3 = cv2.rotate(img[y-f:y+h+f, 0:x], cv2.ROTATE_90_CLOCKWISE)
    img4 = cv2.rotate(img[y-f:y+h+f, x+w:img.shape[1]-1], cv2.ROTATE_90_CLOCKWISE)

    #original = search_dimensions(img1)
    original = search_dimensions(img1)

    return original

def orez_okraj(image):
    img = image.copy()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(img.shape)

    img_area = img.shape[0]*img.shape[1]

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    thresh = cv2.threshold(img,220,255,0)[1]
    thresh = cv2.bitwise_not(thresh) #opecnv countours finds black objects on white bacground
    
    print(thresh.shape)

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    k = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if(w*h > img_area*0.6):
                k.append(cnt)

    if len(k) == 0:
        return thresh

    s = k[0]
    for c in k:
        if cv2.contourArea(c) < cv2.contourArea(s): s = c

    x, y, w, h = cv2.boundingRect(s)
    
    f = 5
    x, y, w, h = x+f, y+f, w-2*f, h-2*f  #scale down by f pixels
    return image[y:y+h, x:x+w]

def detect(image):
    img = image.copy()
    # if len(img.shape) > 2:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # else: image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        
    kernel = np.ones((9, 9), np.uint8)
    img = cv2.erode(img, kernel, iterations=9)

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #thresh = cv2.threshold(img,220,255,0, cv2.THRESH_BINARY_INV)[1]

    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    objekty = []
    for i in range(0, numLabels):
        if i > 0:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 20000:
                objekty.append([x,y,w,h])
                #cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 5)
    #cv2.imshow('',cv2.resize(image, (1000,1000)))
    #cv2.waitKey(0)
    return objekty

def get_stavba(img):
    img = orez_okraj(img)
    original = img.copy()
    #original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

    objekty = detect(img)
    i = 0
    for j,objekt in enumerate(objekty):
        area = objekt[2] * objekt[3]
        if area > objekty[i][2] * objekty[i][3]:
            i = j
        x,y,w,h = objekt
        #cv2.rectangle(original, (x,y), (x+w,y+h), (0,255,0), 3)

    x,y,w,h = objekty[i]
    # return img[y:y+h, x:x+w], cv2.resize(original, IMG_SIZE)
    return img[y:y+h, x:x+w]


