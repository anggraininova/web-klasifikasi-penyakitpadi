from img_classification import teachable_machine_classification
import keras
from PIL import Image, ImageOps
import numpy as np
import boto3

import streamlit as st
from streamlit_option_menu import option_menu

#title z
st.set_page_config(page_title='Image Classification Web App', layout = 'wide')
#menu
with st.sidebar:
	selected = option_menu(
		menu_title="Main menu",
		options=["Home", "Image Classifier", "About"], 

	)

#home
if selected == "Home":
	st.header("Image Classification Web App")
	#st.markdown("<p style='margin-bottom:50px;'>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>", unsafe_allow_html=True)
	#############################################

	st.subheader("Latar Belakang")
	st.markdown("<p style='margin-bottom:50px; text-align:justify;'>Indonesia merupakan negara yang memiliki angka kebutuhan beras yang tinggi. Hal ini mengharuskan petani untuk memproduksi beras dalam angka yang besar dan dengan kualitas yang baik. Salah satu faktor penyebab menurunnya produksi padi adalah penyakit pada tanaman padi. Masing-masing jenis penyakit membutuhkan penanganan yang berbeda, namun tidak semua petani mengetahui jenis penyakit tersebut sehingga memungkinkan terjadinya kesalahan dalam penanganan. Untuk mempermudah petani mengetahui jenis penyakit tersebut sehingga dibuatlah suatu program yang dapat mengidentifikasi penyakit tanaman padi. Daun padi merupakan bagian tubuh padi yang paling mudah untuk mengidentifikasi gejala penyakit yang timbul pada padi. Hal ini disebabkan daun memiliki penampang yang luas dibandingkan bagian tubuh tanaman padi yang lain, sehingga perubahan warna dan bentuk dapat terlihat lebih jelas. Oleh karena itu, daun dapat digunakan sebagai langkah awal deteksi penyakit pada padi.</p>", unsafe_allow_html=True)

	st.subheader("Tujuan")
	st.markdown("<p style='margin-bottom:50px; text-align:justify;'>Bertujuan memberi pembelajaran kepada petani terkait pengklasifikasian penyakit pada tanaman padi dan mengurangi resiko gagal panen yang diakibatkan dari penyakit pada tanaman padi.</p>", unsafe_allow_html=True)

	#############################################
	st.subheader("Kenali penyakit pada daun padi")
	image_sample1 = Image.open('sample/healthy1.jpg')
	image_sample2 = Image.open('sample/hispa4.jpg')
	image_sample3 = Image.open('sample/brownspot6.jpg')
	image_sample4 = Image.open('sample/leafblast9.jpg')
	sample1, sample2, sample3, sample4 = st.columns(4)
	with sample1:
		st.image(image_sample1, caption="contoh healthy")
	with sample2:
		st.image(image_sample2, caption="contoh hispa")
	with sample3:
		st.image(image_sample3, caption="contoh brownspot")
	with sample4:
		st.image(image_sample4, caption="contoh leafblast")
	###############################################

	
#end home













# Download model dari S3
s3 = boto3.client('s3')
s3.download_file('datasetpadi', 'keras_model.h5', 'keras_model.h5')

if selected == "Image Classifier":
	
	st.header("Image Classifier")
	st.markdown("Kenali penyakit daun padi dengan memasukan gambar pada opsi di bawah")
	uploaded_file = st.file_uploader("Choose a image ...", type="jpg")
	if uploaded_file is not None:
	    image = Image.open(uploaded_file)
	    st.image(image, caption='Uploaded image.', use_column_width=True)
	    st.write("")
	    st.write("Classifying...")
	    confidence_score = teachable_machine_classification(image, 'keras_model.h5')
	    label = teachable_machine_classification(image, 'keras_model.h5')
	    if label == 0:
		    st.write("Actual : _BrownSpot")
		    st.write("Predicted : _BrownSpot")
		    st.write(f"Confidence : {teachable_machine_classification.confidence_score}%")
	    elif label == 1:
	        st.write("Actual: _Healthy")
	        st.write("Predicted: _Healthy")
	        st.write(f"Confidence : {teachable_machine_classification.confidence_score}%")
	    elif label == 2:
	    	st.write("Actual: _Hispa")
	    	st.write("Predicted: _Hispa")
	    	st.write(f"Confidence : {teachable_machine_classification.confidence_score}%")
	    elif label == 3:
	    	st.write("Actual: _LeafBlast")
	    	st.write("Predicted: _LeafBlast")
	    	st.write(f"Confidence : {teachable_machine_classification.confidence_score}%")
	    else:
	    	st.error('Gambar tidak dikenal oleh model! harap memberi gambar yang sesuai', icon="🚨")


if selected == "About":
	#st.markdown("<h1 style='color: white; margin-bottom:50px;'>About Image Classification Web App</h1>", unsafe_allow_html=True)
	st.title("About Page")

	st.header("Tentang Kami")
	col1, col2, col3 = st.columns(3)

	with col1:
	   st.markdown("<p>Nova Anggraini (20SA1238)</p>", unsafe_allow_html=True)
	   

	with col2:
	   
	   st.markdown("<p>Universitas Amikom Purwokerto</p>", unsafe_allow_html=True)
	 
	   

	



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
