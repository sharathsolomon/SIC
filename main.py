import streamlit as st
import cv2 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.models.load_model('model.h5', compile=False)

def main():
   st.title("Salt Identification Challenge")

   image = st.file_uploader('Upload the image below')

   predict_button = st.button('Predict on uploaded image')

        
   if predict_button:
     if image is not None:
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        test_img = cv2.imdecode(file_bytes, 1)
        prediction(test_img)
     else:
        st.text('Please upload the image')
        
def prediction(img):
   img = cv2.resize(img,(512,512),interpolation=cv2.INTER_NEAREST)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img = img/255.
  
   fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 20))
   axes[0].set_title('Image', size='large')
   axes[0].axis('off')
   axes[0].imshow(img)
  
   axes[1].set_title('Predicted mask', size='large')
   axes[1].axis('off')
  
   y_pred  = model.predict(img[np.newaxis,:,:,:])
   y_pred = np.squeeze(y_pred,axis=-1)
   axes[1].imshow(np.round(y_pred[0])) 
   st.pyplot(fig)

if __name__ == "__main__":
   main()   
