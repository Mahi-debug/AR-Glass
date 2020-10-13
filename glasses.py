import cv2
from PIL import Image
import numpy 
import streamlit as st

st.write("""Aligning Glasses on Selfie Image Demo""")


def preprocess_glass(glass_img, red_channel, blue_channel, green_channel):
	
	#converting the image from BGR to RGBA
	img = glass_img.convert("RGBA")
	datas = img.getdata()

	newData = []
	for item in datas:
		#Filtering the RGB Channel pixels
		if item[0] > red_channel and item[1] > green_channel and item[2]>blue_channel:
			#Removing the unwanted pixels
			newData.append((255, 255, 255, 0))
		else:
			newData.append(item)
	img.putdata(newData)
	#converting the Image Object to numpy array object
	open_cv_image = numpy.array(img)
	# # Define the structuring element
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	# Apply the opening operation
	opening = cv2.morphologyEx(open_cv_image, cv2.MORPH_OPEN, kernel)

	#Displaying the filtered image using Streamlit
	st.image(img, caption="Filtered Glasses Image", use_column_width=True) 
	return opening


def superimpose(person_img, glasses_img, red_channel, blue_channel, green_channel):

	#Loading the face and eyes detection models.
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
	eyepair_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_mcs_eyepair_big.xml")

	#processing the glasses to convert the background to transparent.
	img_glasses = preprocess_glass(glasses_img, red_channel, blue_channel, green_channel)
	img_glasses_mask = img_glasses[:, :, 3]
	img_glasses = img_glasses[:, :, 0:3]

	#Reading the person image and converting it to gray color
	person_img = numpy.array(person_img)
	gray_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)

	#Detecting the face in the image
	face = face_cascade.detectMultiScale(gray_img, 1.3, 5)
	print(face)

	for (x,y,w,h) in face:

		#Cropping the image according to the face detected
		roi_gray = gray_img[y:y + h, x:x + w]
		roi_color = person_img[y:y + h, x:x + w]

		#Detecting the eyes in the Face detected ROI
		eyepairs = eyepair_cascade.detectMultiScale(roi_gray)
	    
		for (ex, ey, ew, eh) in eyepairs:

			#calculating the boundaries for the glasses.
			x1 = int(ex - ew / 10)-10
			x2 = int((ex + ew) + ew / 10)+10
			y1 = int(ey)-10
			y2 = int(ey + eh + eh / 2)+10

			if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
				continue

	        # Calculate the width and height of the image with the glasses:
			img_glasses_res_width = int(x2 - x1 )
			img_glasses_res_height = int(y2 - y1 ) 

			# Resize the mask to be equal to the region were the glasses will be placed:
			mask = cv2.resize(img_glasses_mask, (img_glasses_res_width, img_glasses_res_height ))

			# Create the invert of the mask:
			mask_inv = cv2.bitwise_not(mask)

			# Resize img_glasses to the desired (and previously calculated) size:
			img = cv2.resize(img_glasses, (img_glasses_res_width, img_glasses_res_height ))

			# Take ROI from the BGR image:
			roi = roi_color[y1:y2, x1:x2]

			# Create ROI background and ROI foreground:
			roi_background = cv2.bitwise_and(roi, roi, mask=mask_inv)
			roi_foreground = cv2.bitwise_and(img, img, mask=mask)


			# Add roi_bakground and roi_foreground to create the result:
			result = cv2.add(roi_background, roi_foreground)

			#uncomment this line and comment the above line to superimpose the image with weighted.
			# result = cv2.addWeighted(roi_background,0.9,roi_foreground,0.1,0)

			# Set the result into the color ROI:
			roi_color[y1:y2, x1:x2] = result
			st.image(roi_color, caption="Output Image", use_column_width=True)
			roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
			cv2.imwrite("results/output.jpg",roi_color)
			



#Taking the Person Image path as input using Streamlit
person_img_path = st.text_input("type person image path... (Supported formats: png, jpg, jpeg)")

if person_img_path != "":
	img = Image.open(person_img_path)

	#Displaying the Image in streamlit
	st.image(img, caption="Person Image", use_column_width=True)
	#Taking the glasses image path as input
	glasses_img_path = st.text_input("Type Glasses image path.....")
	if glasses_img_path != "":
		glass_img = Image.open(glasses_img_path)
		#Displaying the glasses image
		st.image(glass_img, caption="Glasses Image", use_column_width=True)

		#Using Sliders to give the threshold for the filters in RGB Channels
		red_channel = st.slider('Red Channel Filter', 0, 255, 50)
		st.write("You selected", red_channel)
		green_channel = st.slider('Green Channel Filter', 0, 255, 50)
		st.write("You selected", green_channel)
		blue_channel = st.slider('Blue Filter Channel', 0, 255, 50)
		st.write("You selected", blue_channel)

		#Run Button to run the program after giving the parameters.
		run_button = st.button("Run")
		if run_button:
			superimpose(img, glass_img, red_channel, blue_channel, green_channel)
