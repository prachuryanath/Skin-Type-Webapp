import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import preprocessing
import time
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('saved_model', compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
  return model

with st.spinner('Model is being loaded..'):
  model=load_model()

st.title('Skin Classifier')

st.markdown("Welcome to this simple web application that classifies faceskin types.")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)

def predict(image):
    IMAGE_SHAPE = (224, 224,3)
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['Dry_skin','Normal_skin','Oil_skin']

    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'Dry skin':0,
          'Normal skin':0,
          'Oil skin':0
}
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result

if __name__ == "__main__":
    main()
