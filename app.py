import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import cv2
import numpy as np
import tensorflow as tf
import base64
from pygame import mixer
import threading
import time

# Initialize components
model = tf.keras.models.load_model('C:\\Users\\user\\Desktop\\Deep_learning\\projects\\project\\models\\model.h5')
mixer.init()
sound = mixer.Sound('C:\\Users\\user\\Desktop\\Deep_learning\\projects\\project2\\alarm.wav')

face_cascade = cv2.CascadeClassifier('C:\\Users\\user\\Desktop\\Deep_learning\\projects\\project2\\haarcascadefiles\\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('C:\\Users\\user\\Desktop\\Deep_learning\\projects\project2/haarcascadefiles/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('C:\\Users\\user\\Desktop\\Deep_learning\\projects\project2/haarcascadefiles/haarcascade_righteye_2splits.xml')

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
labels = ['Closed', 'Open']

camera_on = False
cap = None
score = 0
lock = threading.Lock()
camera_frame = None
thicc = 2  # Initialize thicc for rectangle thickness control

# Dash App Layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Drowsiness Detection"

app.layout = dbc.Container([
    html.H1("Drowsiness Detection", className="text-center mt-4 mb-4"),
    dbc.Row([
        dbc.Col([
            html.Button("Start Camera", id="start-button", n_clicks=0, className="btn btn-success me-2"),
            html.Button("Stop Camera", id="stop-button", n_clicks=0, className="btn btn-danger"),
            html.Div(id="camera-status", className="mt-2")
        ])
    ]),
    html.Div(id="eye-status", className="text-center text-info mt-2"),
    html.Div([
        html.Img(id="video-feed", style={"width": "100%", "maxWidth": "600px", "marginTop": "20px"}),
        dcc.Interval(id="interval", interval=200, n_intervals=0)
    ]),
    html.Div(id="drowsiness-score", className="text-center text-white mt-4")
])

def preprocess_eye(eye_img):
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_img = cv2.resize(eye_img, (24, 24))
    eye_img = eye_img / 255.0
    eye_img = eye_img.reshape(1, 24, 24, 1)
    return eye_img

def capture_frame():
    global cap, camera_frame
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return
    
    print("📷 Camera successfully opened.")
    
    # Capture frames while the camera is on
    while camera_on:
        ret, frame = cap.read()  # Read a frame from the camera
        
        # If the frame was not captured successfully, break the loop
        if not ret:
            print("❌ Failed to capture frame.")
            break
        
        # Store the captured frame globally for further processing
        camera_frame = frame
        
        # Sleep to simulate a delay in processing (adjust as needed)
        time.sleep(0.03)  # ~30 FPS
        
    cap.release()  # Release the camera when done
    print("📷 Camera released.")

@app.callback(
    Output("video-feed", "src"),
    Output("drowsiness-score", "children"),
    Output("eye-status", "children"),
    Input("interval", "n_intervals"),
)
def update_video(n):
    global score, thicc

    with lock:
        if camera_frame is None:
            return "", "Camera is off. Please start the camera.", ""
        frame = camera_frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame.shape[:2]

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    rpred = [1]
    lpred = [1]
    eye_status = "Unknown"

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        right_eye = reye.detectMultiScale(roi_gray)
        for (rx, ry, rw, rh) in right_eye:
            eye = roi_color[ry:ry + rh, rx:rx + rw]
            eye = preprocess_eye(eye)
            rpred = np.argmax(model.predict(eye), axis=-1)
            break

        left_eye = leye.detectMultiScale(roi_gray)
        for (lx, ly, lw, lh) in left_eye:
            eye = roi_color[ly:ly + lh, lx:lx + lw]
            eye = preprocess_eye(eye)
            lpred = np.argmax(model.predict(eye), axis=-1)
            break

        if rpred[0] == 0 and lpred[0] == 0:
            score += 1
            eye_status = "Eyes Closed"
        else:
            score = max(score - 1, 0)
            eye_status = "Eyes Open"

        cv2.putText(frame, eye_status, (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 10:
        try:
            sound.play()
        except:
            pass
    else:
        sound.stop()
        
    if score < 0:
        score = 0

    cv2.putText(frame, 'Score:' + str(score), (10, height - 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 10:
        try:
            sound.play()
        except:
            pass
        
        # Control rectangle thickness for alarm
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode()

    return f"data:image/jpeg;base64,{frame_base64}", f"Drowsiness Score: {score}", eye_status

@app.callback(
    Output("camera-status", "children"),
    Output("start-button", "disabled"),
    Output("stop-button", "disabled"),
    Input("start-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    prevent_initial_call=True
)
def handle_camera_controls(start_clicks, stop_clicks):
    global camera_on, cap, lock

    ctx = dash.callback_context

    # If no callback triggered, return no update
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "start-button":
        thread_started = False
        with lock:
            if not camera_on:
                camera_on = True
                thread_started = True

        if thread_started:
            threading.Thread(target=capture_frame, daemon=True).start()

        return "Camera started!", True, False

    elif button_id == "stop-button":
        with lock:
            camera_on = False
            
           
        if cap is not None:
            cap.release()
            
            cv2.destroyAllWindows()

        return "Camera stopped!", False, True
    return dash.no_update
    

if __name__ == '__main__':
    app.run(debug=True)
