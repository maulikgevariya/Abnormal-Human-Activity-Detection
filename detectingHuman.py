from keras.models import load_model
import cv2
from IPython.display import clear_output, Image
import base64
from google.colab.patches import cv2_imshow
import numpy as np

model = load_model('model-010.model')  # model get

body_clsfr=cv2.CascadeClassifier('haarcascade_fullbody.xml')  #

source=cv2.VideoCapture('walking.avi') # video pass

labels_dict={1:'Person',0:'Person'}
color_dict={1:(0,255,0),0:(0,255,0)}  
def arrayShow (imageArray):
    ret, png = cv2.imencode('.png', imageArray)
    encoded = base64.b64encode(png)
    return Image(data=encoded.decode('ascii'))    

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bodies=body_clsfr.detectMultiScale(gray,1.3,5)  

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
        
    clear_output(wait=True)
    fm = arrayShow(img)
    display(fm)    
    #cv2_imshow(img)
    #key=cv2.waitKey(0)
   #convert frames into video by using array
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
     break
        
cv2.destroyAllWindows()
source.release()
