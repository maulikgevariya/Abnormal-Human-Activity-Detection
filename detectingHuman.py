from keras.models import load_model
import cv2
from IPython.display import clear_output, Image
import numpy as np

model = load_model('model-030.model')  # model get

body_clsfr=cv2.CascadeClassifier('haarcascade_fullbody.xml')  #

source=cv2.VideoCapture('Running_Final.mp4') # video pass

labels_dict={0:'Playing_Intrument',1:'Running'}
color_dict={0:(0,255,0),1:(23,255,234)}  
while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bodies=body_clsfr.detectMultiScale(gray,1.1,5)  

    for (x,y,w,h) in bodies:
    
        body_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(body_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        acc=round(np.max(result,axis=1)[0]*100,2)
        cv2.putText(img,str(acc),(x+150,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2) 
        
    cv2.imshow('video',img)
    key=cv2.waitKey(1)
    
    if (key==27):
     break
        
cv2.destroyAllWindows()
source.release()
