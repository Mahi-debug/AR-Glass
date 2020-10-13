""" Superimposing Glasses on Selfies"""

Extract the Zip file and keep all the files in one folder Then do follow the next following steps

To Run

In terminal: 

	pip install -r requirements.txt
	streamlit run glasses.py

To add images and frames for testing:

	Add person's images in images/ folder
	Add glasse's images in frames/ folder


In Browser:

	To add the images_path in streamlit:

	1. person's image path: images/mahi.jpeg
	2. glasses image path: frames/frame1.jpg

*** In the code (glasses.py) comment line 93 and uncomment line 96 to get different results since we will be using the weighted superimpose (But It will turn our glasses to Black) ***

use the "Slider" to filter the white and other pixels in the glasses image

Click on "run Button" to run the program 

Check the output in streamlit
** Result images will be saved in "Results/output.png" ** 

Check my results while working on the project in the "results/" folder.

Observation: We can use this to create the frames to detect the ears and nose to place the frames in the sidewards as well. This way person can also check how the frame looks from sidewards. We can use GAN to generate the images with glasses (due to time and computation constraint couldn't implement it). This project has huge potential as we can make the images with Augmented Reality in our application to show how the glasses looks directly on the face and also use recommendation system based on the face sturcuture which we can get from the computer vision techniques to suggest some good combinations of the colors and also designs of the specs. 