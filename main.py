from flask import Flask, redirect, request, jsonify
import cv2 as cv
from generate_id import Generate
import numpy as np
import base64
from flask_cors import CORS
import os
face_cascade = cv.CascadeClassifier(
    "./OpencvCascades/haarcascade_frontalface_default.xml")


face_recognizer = cv.face.LBPHFaceRecognizer_create()
with open("mylockFile.txt","w") as f:
    status=f.write("open");

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resorces={r'/*': {"orgins": '*'}})



def train(Datalist):
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    try:
        # Load the existing model if it exists
        face_recognizer.read('model.yml')
    except cv.error:
        # If the model doesn't exist, create a new one
        pass
    
    image = cv.imread(Datalist[0])
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    images = [gray]
    labels = [int(Datalist[1])]
    labels = np.array(labels)
    
    face_recognizer.update(images, labels)
    face_recognizer.save('model.yml')
    print("Training complete")

            
    return True

@app.route("/", methods=['POST'])
def home():
    if request.method == 'POST':
        image_data = request.json['image']
        email=request.json['email']
        # Decode Base64 string to bytes
        image_bytes = base64.b64decode(image_data)

        # Convert bytes to numpy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decode image array to OpenCV format
        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        height, width = image.shape[:2]
        # Calculate the rotation matrix
        rotation_matrix = cv.getRotationMatrix2D((width/2, height/2), 90, 1)

        # Apply the rotation to the image
        rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
        

        # check if it will detect face
        gray = cv.cvtColor(rotated_image, cv.COLOR_BGR2GRAY)

        # Detect faces in image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangle around detected faces
        FaceID = Generate.ID()
        image_save=f"./Images/{FaceID}.png"    
        cv.imwrite(image_save, rotated_image)
        for (x, y, w, h) in faces:
            cv.rectangle(rotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            FaceID = Generate.ID()
            image_save=f"./Images/{FaceID}.png"    
            cv.imwrite(image_save, rotated_image)
            train_response=train([image_save,FaceID])
        if train_response:
            return jsonify({"info":"success","image_url": image_save,"userId":FaceID})
        else:
           return jsonify({"info":"locked server busy"})
        
        
            
        

        return "return home"





@app.route("/api/rec/faces",methods=['POST','GET'])
def face_train():
    face_recongonation = cv.face.LBPHFaceRecognizer_create()
    face_recongonation.read("./model.yml")
    face_cascade=cv.CascadeClassifier("./OpencvCascades/haarcascade_frontalface_default.xml")
    if request.method == "POST":
        image_data = request.json['image']
        # Decode Base64 string to bytes
        image_bytes = base64.b64decode(image_data)

        # Convert bytes to numpy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decode image array to OpenCV format
        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        height, width = image.shape[:2]
        # Calculate the rotation matrix
        rotation_matrix = cv.getRotationMatrix2D((width/2, height/2), 90, 1)

        # Apply the rotation to the image
        rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))

        # check if it will detect face
        gray = cv.cvtColor(rotated_image, cv.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        label, confidence = face_recongonation.predict(gray)
        print(label)
        
      
    return jsonify({"userID":label,"info":"success"})
        
        
        
        
       


@app.route("/api/face_rec",methods=['POST','GET'])
def face_rec():
    face_recongonation = cv.face.LBPHFaceRecognizer_create()
    if request.method == "POST":
        image = request.json['image']
        name = request.get_json()
        face_recongonation.read(f"./model.yml")
        label, confidence = face_recongonation.predict(image)
        print(label)
        # check the label id in the database #the label is linked to the name of the person


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
