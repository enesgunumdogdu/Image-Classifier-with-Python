# -*- coding: utf-8 -*-
"""

@author: A.Enes Günümdoğdu
me@enesgunumdogdu.com.tr
"""

import cv2
from keras.datasets import cifar10
from tensorflow import keras 
import matplotlib.pyplot as plt 
(train_X,train_Y),(test_X,test_Y)=cifar10.load_data() 
n=6  

from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout
from keras.layers import Flatten 
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

#Convert to float
train_x=train_X.astype('float32')
test_X=test_X.astype('float32')
train_X=train_X/255.0 
test_X=test_X/255.0
train_Y=np_utils.to_categorical(train_Y)
test_Y=np_utils.to_categorical(test_Y) 
num_classes=test_Y.shape[1]

# Adding Layers
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),padding='same',activation='relu',kernel_constraint=maxnorm(3))) 
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))    
model.add(Dropout(0.5)) 
model.add(Dense(num_classes, activation='softmax'))
sgd=SGD(lr=0.01,momentum=0.9,decay=(0.01/25)) 
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.summary()
model.fit(train_X,train_Y,validation_data=(test_X,test_Y),
epochs=10,batch_size=32)  
_,acc=model.evaluate(test_X,test_Y)
print(acc*100)
model.save("model1_cifar_10epoch.h5")   
results={
    0:'aeroplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck'
  }

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
model = load_model('model1_cifar_10epoch.h5')

classes = { 
    0:'aeroplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck' 
    }



## GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI --GUI -- GUI --GUI -- GUI --GUI -- GUI --GUI -- GUI --GUI -- GUI -- GUI -- GUI --
## Creating GUI 
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy

top=tk.Tk()
top.geometry('800x600')
top.title('Image Classifier Project With Cifar10 Dataset')
top.configure(background='#e92929')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

## FRAMES
#------
#♣Creating Left Frame
frame_sol = Frame(top, bg='#5a5a5a')
frame_sol.place(relx=0, rely=0, relwidth=0.5, relheight=0.05)

#Creating Right Frame 
frame_sag = Frame(top, bg='#5a5a5a')
frame_sag.place(relx=0.5, rely=0, relwidth=0.5, relheight=0.05)

##Creating Bottom Frame 
frame_alt = Frame(top, bg='#5a5a5a')
frame_alt.place(relx=0, rely=0.95, relwidth=1, relheight=0.05)

## LABELS
#------
#Creating Label 1
egitim = Label(frame_sol, bg='#5a5a5a', text = "", font ="Verdana 12 bold")
egitim.pack(padx=10, pady=0)

# Creating Label 2 / 'Test' 
test = Label(frame_sag, bg='#5a5a5a', text = "", font ="Verdana 12 bold")
test.pack(padx=10, pady=0)

#Creating Label 3 / 
projeyi_hazirlayan = Label(frame_alt, bg='#5a5a5a', text = "A.Enes Günümdoğdu-----enesgunumdogdu0@gmail.com",font ="Quicksand 12 bold" )
projeyi_hazirlayan.pack(padx=10, pady=5, side=BOTTOM)

# FUNCTIONS
#------
#Creating Camera Function // Unactive for now.
def buton_fonksiyonu():
    import numpy as np
    import cv2
    
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        
    cap.release()
    cv2.destroyAllWindows()

# Classifier Function 
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#FF0000', text=sign,background='#000000',font ="Verdana 20 bold")
    label.pack(padx=10,pady=10,side=RIGHT,anchor=NE)

# Classify Button Function 
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify the Image",command=lambda: classify(file_path),padx=10,pady=10)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.01,rely=0.3,anchor=W)

#  Upload Image Function 
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
    
def data_msg():
    import tkinter.messagebox
    tkinter.messagebox.showinfo("Data Information",  "Our data includes 60000 32*32res image for each class. 500000 train image and 10000 test image.")    
    
# BUTTONS
#------
#Creating Button 2 / 'Upload Image'
upload=Button(top,text="Upload Image",command=upload_image,padx=25,pady=10)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=TOP,anchor=NW,padx=10,pady=50)

#Creating Button 3 / 'Get Data Information' 
data_bilgi=Button(top,text="Get Data Information",command=data_msg,padx=25,pady=10)
data_bilgi.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
data_bilgi.pack(side=TOP, anchor=NE,padx=10,pady=50)
data_bilgi.place(x=550, y=55)
    

sign_image.pack(anchor=W,side=LEFT,expand=True)
label.pack(anchor=NW,padx=10,pady=50)
top.mainloop()
