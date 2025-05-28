from pathlib import Path
import pathlib
import streamlit as st
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from fastai.vision.all import load_learner, PILImage


# Windowsda model yuklash uchun bu muhim


st.title('Transport tasniflash modeli')

file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'jpg'])

# Avval tekshiramiz, fayl bor-yo‘qligini
if file is not None:
    st.image(file)  # rasmni ko‘rsatish
    st.success('Rasm yuklandi, model ishlayapti...')

    # Rasmni PILImage formatiga o‘tkazamiz
    img = PILImage.create(file.getvalue())

    #Modelni yuklaymiz
    model = load_learner('transport_model.pkl')

    #Bashorat qilamiz
    pred, prob_id, probs = model.predict(img)

    #Natijani chiqaramiz
    st.success(f"Model bashorati: {pred}")
    st.info(f"Ehtimol: {probs[prob_id]:.3%}")
    # st.image(img.to_thumb(256, 256))
else:
    st.info("Iltimos, rasm yuklang.")