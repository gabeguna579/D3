import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
from twilio.rest import Client
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import pyautogui
import time
import boto3
SCREEN_SIZE=(1920,1080)
fourcc=cv2.VideoWriter_fourcc(*"XVID")
out=cv2.VideoWriter('drowsiness.mp4',fourcc,20.0,(SCREEN_SIZE))
fps=120
prev=0
smtp_server = 'smtp.gmail.com'
smtp_port = 587
smtp_username = 'xxxxxxxxxx@gmail.com'
smtp_password = 'xxxxxxxxxxx'
receiving_email = 'xxxxxxxxxxxx@gmail.com'
subject = 'Driver who is sleeping'
msg = MIMEMultipart()
msg['From'] = smtp_username
msg['To'] = receiving_email
msg['Subject'] = subject
account_sid = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
auth_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'
client = Client(account_sid, auth_token)
to_number = '+9191xxxxxxxx'
from_number = '+12xxxxxxxxxxxxxxxxxx'
twiml_url = 'http://demo.twilio.com/docs/voice.xml'
message = 'Sleeping for about 60 sec'
mixer.init()
sound = mixer.Sound('alarm.wav')
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
model = load_model('models/model.h5')
path = os.getcwd()
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_COMPLEX
count = 0
score = 0
rpred = [99]
lpred = [99]
notification=0
while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    time_elapsed = time.time() - prev
    img = pyautogui.screenshot()
    if time_elapsed > 1.0 / fps:
        prev = time.time()
        frame1 = np.array(img)
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        break
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        break
    if (rpred[0] == 0 and lpred[0] == 0):
        print(score)
        score = score + 1
        cv2.putText(frame, "Drowsy", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        print(score)
        score = 0
        cv2.putText(frame, "", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score < 0):
        score = 0
    if (score > 5):
        try:
            sound.play()
        except:
            pass
        cv2.imwrite(os.path.join(path, 'deep_learning.jpg'), frame)
        if (score > 10):
            notification+=1
            if (notification % 40) == 0:
                out.release()
                client_s3 = boto3.client('s3', aws_access_key_id='xxxxxxxxxxxxxxxxxxxx',
                                      aws_secret_access_key='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                bucket = 'xxxxxxxxxxxxxxxxxxxx'
                cur_path = os.getcwd()
                file = 'drowsiness.mp4'
                filename = os.path.join(cur_path, file)
                client_s3.upload_file(filename, bucket, file)
                url = 'd7wxug00sdmmf.cloudfront.net/' + file
                print(f"Use this url to see the video {url}")
                call = client.calls.create(
                to=to_number,
                from_=from_number,
                url=twiml_url)
                message = client.messages.create(
                body=message,
                from_=from_number,
                to=to_number
                )
                with open('deep_learning.jpg', 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data, name='deep_learning')
                    msg.attach(image)
                smtp_session = smtplib.SMTP(smtp_server, smtp_port)
                smtp_session.starttls()
                smtp_session.login(smtp_username, smtp_password)
                smtp_session.sendmail(smtp_username, receiving_email, msg.as_string())
                smtp_session.quit()
                print('Captured image was successfully sent to the mail ID', receiving_email)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()