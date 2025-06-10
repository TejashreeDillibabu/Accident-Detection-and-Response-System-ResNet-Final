from flask import Flask, render_template, Response,request, send_from_directory, send_file, redirect, url_for
import sqlite3
import cv2
import tensorflow as tf 
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
import easygui
import cv2
from detection import DetectionModel
import numpy as np
import os
import numpy as np
from PIL import Image
import cv2
import math
import time
from datetime import datetime
import smtplib
import base64
import threading


tf.compat.v1.enable_eager_execution()

output_folder = "accident_captures"  # Folder to save captured images
os.makedirs(output_folder, exist_ok=True) 

model = DetectionModel("model.json", 'model_weights.keras')
from tensorflow.keras.preprocessing import image

def preprocess_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    img = img.resize((224, 224))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    
    img_array = img_array / 255.0
    
    return img_array

def predict_image_class(image_array):
    class_labels = ['Major Accident Detected', "Minor Accident Detected", 'No Accident Detected']
    
    # Get prediction from the model
    result = model.predict(image_array)
    
    # Get the index of the highest probability class
    predicted_class_index = np.argmax(result)
    
    # Return the corresponding class label
    predicted_class = class_labels[predicted_class_index]
    return predicted_class



app=Flask(__name__)

database="final2.db"

def createtable():
    conn=sqlite3.connect(database)
    cursor=conn.cursor()
    cursor.execute("create table if not exists policereg(id integer primary key autoincrement,regno text, password text)")
    cursor.execute("CREATE TABLE IF NOT EXISTS accident_data (id integer primary key autoincrement, accident_type TEXT, person_count INTEGER, timestamp TEXT, image_path TEXT, image BLOB, status TEXT)")

    cursor.execute("create table if not exists staff(id integer primary key autoincrement, StaffId text, StaffName text, password text)")
    cursor.execute("create table if not exists admin(id integer primary key autoincrement, StaffId text, password text)")
    conn.commit()
    conn.close()
createtable()

font = cv2.FONT_HERSHEY_SIMPLEX
@app.route('/')


@app.route('/index')
def index():
    return render_template('index.html')

def convert_image_to_binary(image_path):
    with open(image_path, 'rb') as file:
        binary_data = file.read()
    return binary_data

import base64

def base64_encode(data):
    # Ensure data is in bytes
    if isinstance(data, str):
        data = data.encode('utf-8')  # Convert string to bytes
    return base64.b64encode(data).decode('utf-8')


classNames = []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: 
        objects = classNames

    objectInfo = []
    person_count = 0  # Initialize person count

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    # Draw bounding box and label
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                if className == 'person':  # If the detected class is "person"
                    person_count += 1  # Increase person count

    return img, objectInfo, person_count

output_folder = "static/accident_images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def convert_image_to_binary(image_path):
    with open(image_path, "rb") as file:
        return file.read()
import cv2
import time
import os
import sqlite3
import numpy as np
from datetime import datetime

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Unable to open the video file."
    
    accident_detected = False
    accident_type = "No Accident"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    final_person_count = 0
    frame_skip = 5  # Process every 5th frame for speed
    frame_count = 0
    accident_frames = []
    person_counts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when the video ends

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames for faster processing

        # **Accident Detection**
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (224, 224))
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        prob = round(prob[0][0] * 100, 2)

        if pred in ['Major Accident Detected', 'Minor Accident Detected']:
            accident_detected = True
            accident_type = pred
            accident_frames.append(frame)

        # **Person Detection**
        _, _, person_count = getObjects(frame, 0.45, 0.2, objects=['person'])
        person_counts.append(person_count)

        cv2.imshow('Processed Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # **If an accident was detected, save an image**
    image_filename = None
    image_data = None
    if accident_detected and accident_frames:
        image_filename = os.path.join(output_folder, f"accident_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(image_filename, accident_frames[-1])  # Save the last accident frame
        image_data = convert_image_to_binary(image_filename)

    # **Update the database after processing the full video**
    final_person_count = max(person_counts) if person_counts else 0  # Get max person count
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO accident_data (accident_type, person_count, timestamp, image_path, image, status)
        VALUES (?, ?, ?, ?, ?, 0)
    """, (accident_type, final_person_count, timestamp, image_filename, image_data))
    conn.commit()
    conn.close()

    return redirect(url_for('index'))
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No file part"
        
        file = request.files['video']
        if file.filename == '':
            return "No selected file"
        
        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)
        
        return process_video(file_path)
    
    return render_template('upload.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Unable to open the camera stream."
    
    accident_detected = False
    person_detection_started = False
    start_time = 0
    final_person_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (224, 224))
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        prob = round(prob[0][0] * 100, 2)
        
        if pred == 'Major Accident Detected' or pred == 'Minor Accident Detected':
            accident_type = pred
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow('CCTV CAMERA', frame)
            
            # Save image
            image_filename = os.path.join(output_folder, f"accident_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(image_filename, frame)
            print(f"Image saved: {image_filename}")

            image_data = convert_image_to_binary(image_filename)
            
            
            start_time = time.time()
            person_detection_started = True
        
        # Person detection logic for 1 minute
        if person_detection_started:
            elapsed_time = time.time() - start_time
            if elapsed_time <= 60:  # Run person detection for 1 minute
                result, objectInfo, person_count = getObjects(frame, 0.45, 0.2, objects=['person'])
                final_person_count = person_count  # Keep updating person count during this period

                # Display the output with person count
                cv2.putText(result, f'Person Count: {person_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("CCTV CAMERA", result)
                print(f"Detected Persons: {person_count}")
            else:
                # After 1 minute, save data to the database and reset detection flags
                print(f"Final Person Count: {final_person_count}")
                conn = sqlite3.connect(database)
                cursor = conn.cursor()
                cursor.execute("INSERT INTO accident_data (accident_type, person_count, timestamp, image_path, image, status) VALUES (?, ?, ?, ?, ?, 0)",
                   (accident_type, person_count, timestamp, image_filename, image_data))
                conn.commit()
                conn.close()
                person_detection_started = False  # Stop further person detection
                accident_detected = False
        
        
        cv2.putText(frame, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('CCTV CAMERA', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('index'))

a=[]
@app.route('/register', methods=["GET","POST"])
def register():    
    if request.method=="POST":
        regno=request.form['regno']     
        
        password=request.form['password']
        conn=sqlite3.connect(database)
        cursor=conn.cursor()
        cursor.execute(" SELECT regno FROM policereg WHERE regno=?",(regno,))
        registered=cursor.fetchall()
        if registered:
            return render_template('index.html',show_alert1=True)
        else:
            cursor.execute("insert into policereg(regno,password) values(?,?)",(regno,password))
            conn.commit()
            return render_template('index.html', show_alert2=True)
    return render_template('index.html')


@app.route("/approve_pol",methods=['POST','GET'])
def approve_pol():
    idnum=request.form['idnum']
    conn=sqlite3.connect(database)
    cursor=conn.cursor()
    cursor.execute("update accident_data set status=? where id =?",(1,idnum))
    conn.commit()
    conn=sqlite3.connect(database)
    cursors=conn.cursor()
    cursors.execute("select * from accident_data where status=0")
    data=cursors.fetchall()
    conn.commit()
    cursors.execute("select * from accident_data where status=1")
    data1=cursors.fetchall()
    conn.commit()
    return render_template('upload_ass.html', data1=data, data3=data1, base64_encode=base64_encode)

@app.template_filter('b64encode')
def base64_encode(data):
    return base64.b64encode(data).decode('utf-8')

def send_email():
    # Configure your email settings
    sender_email = "y59832071@gmail.com"
    receiver_email = "210701287@rajalakshmi.edu.in"
    password = "opptdadldnkjhzzi"
    message = "An accident has been detected. Please respond immediately."

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
        print(f"Error sending email: {e}")

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_emails(recipient, subject, message):
    sender_email = "y59832071@gmail.com"
    sender_password = "opptdadldnkjhzzi"

    # Set up the email details
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = subject

    # Attach the message
    msg.attach(MIMEText(message, 'plain'))

    # Set up the server and send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, msg.as_string())
        server.quit()
        print(f"Email sent to {recipient}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Example of how you can send two emails
def send_alert_emails():
    # First email details
    recipient1 = "210701287@rajalakshmi.edu.in"
    subject1 = "Alert: Accident Detected"
    message1 = "An accident has been detected. Please respond immediately."

    # Send the first email
    send_emails(recipient1, subject1, message1)

    # Second email details
    recipient2 = "210701317@rajalakshmi.edu.in"
    subject2 = "Alert: Medical Assistance Needed"
    message2 = "An accident has been detected. Medical assistance is required."

    # Send the second email
    send_emails(recipient2, subject2, message2)



a=[]
sname=[]
Sid=[]
sec=[]
@app.route('/login', methods=["GET","POST"])
def login():
    global regno
    if request.method == "POST":
        regno=request.form['regno']
        print(regno)
        password=request.form['password']
        print(password)
        conn=sqlite3.connect(database)
        cursor=conn.cursor()
        cursor.execute("select * from policereg where regno=? AND password=?",(regno,password))
        data=cursor.fetchone()
        
        print(a)
        if data is None:
            return render_template('index.html', show_alert4=True)
        else:            
            conn=sqlite3.connect(database)
            cursor=conn.cursor()
            cursor.execute("select * from policereg where regno=?",(regno,))
            results=cursor.fetchone()
            conn.commit()
            conn=sqlite3.connect(database)
            cursors=conn.cursor()
            cursors.execute("select * from accident_data where status=0")
            data=cursors.fetchall()
            conn.commit()
            cursors.execute("select * from accident_data where status IN (1,2)")
            data1=cursors.fetchall()
            conn.commit()
            #print(data)
            return render_template('upload_ass.html', data1=data, data3=data1, base64_encode=base64_encode)
    return render_template('login.html')


def update_status():
                    # Adjust the format according to your DB's timestamp format
    import datetime
    import time
    while True:
        # Fetch records with status 0
        conn=sqlite3.connect(database)
        cursors=conn.cursor()
        cursors.execute("SELECT id, timestamp, accident_type FROM accident_data WHERE status = 0")
        records = cursors.fetchall()
        

        for record in records:
            record_time_str = record[1]  # Assuming the timestamp is in the second column
            current_time = datetime.datetime.now()

            # Convert the string timestamp to a datetime object (adjust format if necessary)
            try:

                record_time = datetime.datetime.strptime(record_time_str, '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(f"Error parsing timestamp: {e}")
                continue

            # Calculate the time difference
            time_difference = current_time - record_time

            # Check if 5 minutes have passed
            if time_difference.total_seconds() >= 300:  # 300 seconds = 5 minutes
                # Update the status to 1
                #cursors.execute("UPDATE accident_data SET status = 1 WHERE id = %s", (record[0],))  # Assuming the id is in the first column
                #conn.commit()
                if record[1] == 'Minor Accident Detected':
                    send_email()  # Send the email notification
                else:
                    send_alert_emails()
                    

        time.sleep(60)


@app.route('/upload_ass', methods=["GET","POST"])
def upload_ass():
    return render_template('upload_ass.html')



@app.route("/approve_hos",methods=['POST','GET'])
def approve_hos():
    idnum=request.form['idnum']
    conn=sqlite3.connect(database)
    cursor=conn.cursor()
    cursor.execute("update accident_data set status=? where id =?",(2,idnum))
    conn.commit()
    conn=sqlite3.connect(database)
    cursors=conn.cursor()
    cursors.execute("select * from accident_data where status=0 where accident_type = 'Major Accident Detected'")
    data=cursors.fetchall()
    conn.commit()
    cursors.execute("select * from accident_data where status=2")
    data1=cursors.fetchall()
    conn.commit()
    return render_template('hospital.html', data1=data, data3=data1, base64_encode=base64_encode)

@app.route('/service',methods=["GET","POST"])
def service():
    if request.method=='POST':
        StaffId=request.form['StaffId']        
        StaffName=request.form['StaffName']
        password=request.form['password']
        conn=sqlite3.connect(database)
        cursor=conn.cursor()
        cursor.execute("SELECT StaffId from staff WHERE StaffId=?",(StaffId,))
        registered=cursor.fetchall()
        if registered:
            return render_template('index.html',show_alert3=True)
        else:
            cursor.execute("INSERT INTO staff(StaffId,StaffName,password) VALUES (?,?,?)",(StaffId,StaffName,password))
            conn.commit()
            return render_template('index.html',show_alert2=True)

@app.route('/admin',methods=["GET","POST"])
def admin():
    if request.method=='POST':
        StaffId=request.form['StaffId']     
       
        password=request.form['password']
        conn=sqlite3.connect(database)
        cursor=conn.cursor()
        cursor.execute("SELECT StaffId from admin WHERE StaffId=?",(StaffId,))
        registered=cursor.fetchall()
        if registered:
            return render_template('index.html',show_alert3=True)
        else:
            cursor.execute("INSERT INTO admin(StaffId,password) VALUES (?,?)",(StaffId,password))
            conn.commit()
            return render_template('index.html',show_alert2=True)



staf_sec=[]
@app.route('/service_login', methods=["GET","POST"])
def service_login():
    if request.method == "POST":
        StaffId=request.form['StaffId']
        password=request.form['password']
        conn=sqlite3.connect(database)
        cursor=conn.cursor()
        cursor.execute("select * from staff where StaffId=? AND password=?",(StaffId,password))
        data=cursor.fetchone()
        if data is None:
            return render_template('index.html', show_alert4=True)
         
        
        else:
            conn=sqlite3.connect(database)
            cursors=conn.cursor()
            cursors.execute("select * from accident_data where status=0 and accident_type = 'Major Accident Detected'")
            data=cursors.fetchall()
            conn.commit()
            cursors.execute("select * from accident_data where status=2")
            data1=cursors.fetchall()
            conn.commit()
            #print(data)
            return render_template('hospital.html', data1=data, data3=data1, base64_encode=base64_encode)
            


@app.route('/admin_login', methods=["GET","POST"])
def admin_login():
    if request.method == "POST":
        username=request.form['username']
        password=request.form['password']        
        if username== "admin" and password=="admin":
                conn=sqlite3.connect(database)
                cursor=conn.cursor()
                cursor.execute(" SELECT *  FROM accident_data WHERE status=?",(0,))
                data=cursor.fetchall()
                cursor.execute(" SELECT *  FROM accident_data WHERE status IN (1,2,3)")
                data1=cursor.fetchall()
                print(type(data1[0][6]))
                cursor.execute(" SELECT *  FROM accident_data WHERE status=?",(3,))
                data2=cursor.fetchall()
                return render_template('admin_view.html',data=data,data1=data1,data2=data2,base64_encode=base64_encode)
        else:
                return render_template('index.html')
    return render_template('index.html')





if __name__ == "__main__":
    thread = threading.Thread(target=update_status)
    thread.start()
    app.run()
