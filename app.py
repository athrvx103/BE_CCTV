from flask import Flask, render_template, request, redirect, url_for
import cv2
import time
from datetime import datetime
import numpy as np
from collections import deque
from keras.models import load_model
import telepot

app = Flask(__name__)

# Initialize the Telegram bot with your bot's API token
bot = telepot.Bot('7559235446:AAEUoodtqDTWWvQh8tvd4N7w6FQtUapFSbA')  # Replace with your actual token

# def save_annotated_video(input_video, output_video, telegram_group_id):
#     print("Loading model ...")
#     model = load_model('modelnew.h5')
#     Q = deque(maxlen=128)
    
#     # Check if the input_video is an integer (webcam) or a filename
#     if isinstance(input_video, int): #Here we check if the input is a webcam or a video file ie if it is of type int then it is a webcam
#         vs = cv2.VideoCapture(input_video)
#         webcam = True
#     else:
#         vs = cv2.VideoCapture(input_video)
#         webcam = False
    
#     (W, H) = (None, None)
#     violence_detected = False
#     violence_start_frame = None
#     frame_count = 0
    
#     # Define the codec and create a VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video, fourcc, 30.0, (1280, 720))  # Adjust the resolution as needed

#     smoothing_window = 10  # Adjust the window size for smoothing
#     prediction_history = deque(maxlen=smoothing_window)

#     # Start the timer
#     start_time = time.time()

#     while True:
#         (grabbed, frame) = vs.read()

#         if not grabbed:
#             break

#         if W is None or H is None:
#             (H, W) = frame.shape[:2]

#         output = frame.copy()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = cv2.resize(frame, (128, 128)).astype("float32")
#         frame = frame.reshape(128, 128, 3) / 255

#         preds = model.predict(np.expand_dims(frame, axis=0))[0]
#         Q.append(preds)

#         results = np.array(Q).mean(axis=0)
#         i = (preds > 0.50)[0]
#         prediction_history.append(i)

#         smoothed_prediction = np.mean(prediction_history) > 0.5
#         label = smoothed_prediction

#         text_color = (0, 255, 0) 

#         if label:
#             text_color = (0, 0, 255)
            
#             if not violence_detected:
#                 violence_detected = True
#                 violence_start_frame = frame_count
#                 violence_start_time = time.time()
#         else:
#             violence_detected = False

#         if violence_detected and frame_count == violence_start_frame + 10:
#             # Capture the current timestamp
#             current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#             # Send the alert with timestamp to the Telegram group
#             message = f"Violence detected at {current_time}"
#             with open('alert_frame.jpg', 'wb') as f:
#                 cv2.imwrite('alert_frame.jpg', frame * 255)
#                 bot.sendPhoto(telegram_group_id, open('alert_frame.jpg', 'rb'), caption=message)

#         text = "Violence: {}".format(label)
#         FONT = cv2.FONT_HERSHEY_SIMPLEX

#         cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

#         # Write the frame with annotations to the output video
#         out.write(output)

#         cv2.imshow("Violence Detection", output)

#         key = cv2.waitKey(1) & 0xFF
#         frame_count += 1

#         # Stop the webcam after 10 seconds
#         elapsed_time = time.time() - start_time
#         if elapsed_time > 10:
#             print("[INFO] Stopping webcam after 10 seconds.")
#             break

#         if key == ord("q"):
#             print("[INFO] User requested to stop the webcam.")
#             break

#     print("[INFO] Cleaning up...")
#     vs.release()
#     out.release()  # Release the output video writer
#     cv2.destroyAllWindows()

#     # Send the annotated video to the Telegram group
#     print("[INFO] Sending annotated video to Telegram...")
#     with open(output_video, 'rb') as video_file:
#         bot.sendVideo(telegram_group_id, video_file, caption="Annotated video of detected violence.")

def save_annotated_video(input_video, output_video, telegram_group_id):
    print("Loading model ...")
    model = load_model('modelnew.h5')
    Q = deque(maxlen=128)

    # Open the video source
    vs = cv2.VideoCapture(input_video)
    fps = int(vs.get(cv2.CAP_PROP_FPS))  # Get the frame rate of the video
    frames_for_20_seconds = fps * 20  # Number of frames corresponding to 20 seconds

    (W, H) = (None, None)
    violence_detected = False
    violent_frame_count = 0  # Counter for violent frames
    violence_threshold = 20  # Number of violent frames required to send an alert

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (1280, 720))  # Adjust the resolution as needed

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        i = (preds > 0.50)[0]

        if i:
            if not violence_detected:
                # First detection of violence
                violence_detected = True
                violent_frame_count = 1  # Start counting violent frames
                start_time = time.time()  # Record the start time of detection
            else:
                # Increment the counter for violent frames
                violent_frame_count += 1
        else:
            # Reset everything if violence is not detected
            violence_detected = False
            violent_frame_count = 0

        # Check if violence has been detected for more than the threshold within 20 seconds
        elapsed_time = time.time() - start_time if violence_detected else 0
        if violence_detected and violent_frame_count >= violence_threshold and elapsed_time <= 20:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Send the alert with timestamp to the Telegram group
            message = f"Violence detected for {violent_frame_count} frames within 20 seconds at {current_time}"
            with open('alert_frame.jpg', 'wb') as f:
                cv2.imwrite('alert_frame.jpg', frame * 255)
                bot.sendPhoto(telegram_group_id, open('alert_frame.jpg', 'rb'), caption=message)

            # Reset the counter and detection flag after sending the alert
            violent_frame_count = 0
            violence_detected = False

        text = "Violence: {}".format(i)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, (0, 0, 255) if i else (0, 255, 0), 3)

        # Write the frame with annotations to the output video
        out.write(output)

        cv2.imshow("Violence Detection", output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] User requested to stop the webcam.")
            break

    print("[INFO] Cleaning up...")
    vs.release()
    out.release()  # Release the output video writer
    cv2.destroyAllWindows()

    # Send the annotated video to the Telegram group
    # print("[INFO] Sending annotated video to Telegram...")
    # with open(output_video, 'rb') as video_file:
    #     bot.sendVideo(telegram_group_id, video_file, caption="Annotated video of detected violence.")   
        
        
        
def save_annotated_video2(input_video, telegram_group_id):
    print("Loading model ...")
    model = load_model('modelnew.h5')
    Q = deque(maxlen=128)

    vs = cv2.VideoCapture(input_video)
    (W, H) = (None, None)
    violence_detected = False
    consecutive_violence_frames = 0  # Counter for consecutive violent frames
    violence_threshold = 10  # Number of frames after the first detection to send an alert

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter('annotated_video.avi', fourcc, 30.0, (1920, 1080))

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame / 255.0

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        i = (preds > 0.50)[0]

        if i:
            if not violence_detected:
                # First detection of violence
                violence_detected = True
                consecutive_violence_frames = 0  # Reset the counter for consecutive frames
            else:
                # Increment the counter for consecutive violent frames
                consecutive_violence_frames += 1
        else:
            # Reset everything if violence is not detected
            violence_detected = False
            consecutive_violence_frames = 0

        # Check if violence has been detected for more than the threshold after the first detection
        if violence_detected and consecutive_violence_frames >= violence_threshold:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            frame_image = (frame * 255).astype(np.uint8)  # Save the frame image

            # Send the alert with timestamp to the Telegram group
            message = f"Violence detected for more than {violence_threshold} frames at {current_time}"
            with open('alert_frame.jpg', 'wb') as f:
                cv2.imwrite('alert_frame.jpg', frame_image)
                bot.sendPhoto(telegram_group_id, open('alert_frame.jpg', 'rb'), caption=message)

            # Reset the counter after sending the alert
            consecutive_violence_frames = 0

        text = "Violence: {}".format(i)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, (0, 0, 255) if i else (0, 255, 0), 3)

        output_video.write(output)

        cv2.imshow("Crime Detection", output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    print("[INFO] Cleaning up...")
    vs.release()
    output_video.release()
    cv2.destroyAllWindows()

    return frame_image if violence_detected else None

def convert_to_hd_black_and_white(frame_image):
    frame_bw = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)

    frame_hd = cv2.resize(frame_bw, (1920, 1080), interpolation=cv2.INTER_CUBIC)

    return frame_hd

def send_frame_to_telegram(frame_image, telegram_group_id):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frame_message = f"Violence detected at {current_time} "

    frame_bw = convert_to_hd_black_and_white(frame_image)
    frame_filename = 'frame_image_bw.jpg'
    cv2.imwrite(frame_filename, frame_bw)
    with open(frame_filename, 'rb') as f:
        bot.sendPhoto(telegram_group_id, f, caption=frame_message)
    # os.remove(frame_filename)

# Define the route to the homepage
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')
# @app.route('/index')
# def input_page():
#     return render_template('index.html')

@app.route('/choose_video',methods=['GET', 'POST'])
def index():
    return render_template('choose_video.html')

@app.route('/choose_method', methods=['GET', 'POST'])
def method():
    if request.method == 'POST':
        source = request.form.get('source')

        if source == 'webcam':
            # Start the webcam detection process
            input_video = 0  # Webcam source
            output_video_file = 'annotated_video.avi'
            telegram_group_id = '-1002674875201'  # Replace with your Telegram group ID
            save_annotated_video(input_video, output_video_file, telegram_group_id)
            return render_template('TelegramUpdate.html')

        elif source == 'video':
            # Redirect to the page for uploading a video
            print("hiiee")
            return redirect(url_for('index'))

    return render_template('choose_method.html')

# Define a route to process the form submission
@app.route('/detect_crime', methods=['POST'])
def detect_crime():
  # Check if a file is part of the request
    if 'video_file' not in request.files:
        return "No video file uploaded.", 400

    video_file = request.files['video_file']

    # Check if the user selected a file
    if video_file.filename == '':
        return "No selected file.", 400

    # Save the uploaded video file
    video_path = f"uploaded_videos/{video_file.filename}"
    video_file.save(video_path)

    # Process the uploaded video
    output_video_file = 'annotated_video.avi'
    telegram_group_id = '-1002674875201'  # Replace with your Telegram group ID
    input_video = 'IMG_3138.mp4'
    telegram_group_id = '-1002674875201'
    
    frame_image = save_annotated_video2(video_path, telegram_group_id)
    
    if frame_image is not None:
        send_frame_to_telegram(frame_image, telegram_group_id)
    # save_annotated_video(video_path, output_video_file, telegram_group_id)

    # Redirect to a page that informs the user to check Telegram
    return render_template('TelegramUpdate.html')
if __name__ == '__main__':
    app.run(debug=True)
