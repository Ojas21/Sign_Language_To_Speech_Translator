import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import keras
import keras
import keras.utils
from keras import utils as np_utils

#pre-trained model used
model = keras.models.load_model("asl_classifier.h5")
#dictionary of the alphabets which algorithm can detect
labels_dict = {0:'0', 
                 1:'A', 
                 2:'B', 
                 3:'C', 
                 4:'D', 
                 5:'E',
                 6:'F',
                 7:'G',
                 8:'H',
                 9:'I',
                 10:'J',
                 11:'K',
                 12:'L',
                 13:'M',
                 14:'N',
                 15:'O',
                 16:'P',
                 17:"Q",
                 18:'R',
                 19:'S',
                 20:'T', 
                 21:'U', 
                 22:'V',
                 23:'W',
                 24:'X',
                 25:'Y',
                 26:'Z'}
color_dict=(0,255,0)
x=0
y=0
w=64
h=64

import numpy as np
img_size=128
minValue = 70
source=cv2.VideoCapture(0)# used for video capturing
count = 0
string = " "#this variable shows text on screen1
prev = " "
prev_val = 0

while(True):
    ret,img=source.read()

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #cv2.rectangle(img,(x,y),(x+w,y+h),color_dict,2)

    cv2.rectangle(img,(24,24),(250 , 250),color_dict,2)

    crop_img=gray[24:250,24:250]
    count = count + 1

    if(count % 100 == 0):
        prev_val = count

    cv2.putText(img, str(prev_val//100), (300, 150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2) #times color

    blur = cv2.GaussianBlur(crop_img,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    resized=cv2.resize(res,(img_size,img_size))

    normalized=resized/255.0

    reshaped=np.reshape(normalized,(1,img_size,img_size,1))

    result = model.predict(reshaped)
    #print(result)

    label=np.argmax(result,axis=1)[0]
    #counter
    if(count == 300):
        count = 99
        prev= labels_dict[label] 
        if(label == 0):
               string = string + " "
            #if(len(string)==1 or string[len(string)] != " "):
             
        else:
                string = string + prev
    #this code shows the text onscreen
    cv2.putText(img, prev, (24, 14),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) #This snippet showcases alphabet 
    cv2.putText(img, string, (275, 50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)# onscreen in realtime
    cv2.imshow("Gray",res)    
    cv2.imshow('LIVE',img)
    #this snippet closes the application
    key=cv2.waitKey(1)
    
 
    if(key==27):#press Esc. to exit
        
        break
print(string)        

cv2.destroyAllWindows()
source.release()

cv2.destroyAllWindows()

from gtts import gTTS 
from win32com.client import Dispatch

def speak(a):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.speak(a)

option=input("enter 1.single or 2.entire word")

if option=="1":
    for i in string:
        speak(i)
elif option=="2":
        speak(string)




"""  
# This module is imported so that we can  
# play the converted audio 
import os 
  
# The text that you want to convert to audio 
  
# Language in which you want to convert 
language = 'en'
# Passing the text and language to the engine,  
# here we have marked slow=False. Which tells  
# the module that the converted audio should  
# have a high speed 
myobj = gTTS(text=string, lang=language, slow=False) 
  
# Saving the converted audio in a mp3 file named 
# welcome  
myobj.save("welcome2121.mp3") 
  
# Playing the converted file 
os.system("welcome2121.mp3") 




from playsound import playsound
playsound('welcome2121.mp3')
"""