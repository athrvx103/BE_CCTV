import numpy as np
import argparse
import cv2
import os
from keras.models import load_model
from collections import deque
import telepot
import time
from datetime import datetime

# Initialize the Telegram bot with your bot's API token
bot = telepot.Bot('7559235446:AAEUoodtqDTWWvQh8tvd4N7w6FQtUapFSbA')  # Replace 'YOUR_BOT_API_TOKEN' with your actual token

def save_annotated_video(input_video, output_video, telegram_group_id):
    print("Loading model ...")
    model = load_model('modelnew.h5')
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture(input_video)
    (W, H) = (None, None)
    violence_detected = False
    violence_start_frame = None
    frame_count = 0
    
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
        Q.append(preds)

        results = np.array(Q).mean(axis=0)
        i = (preds > 0.50)[0]
        label = i

        text_color = (0, 255, 0) 

        if label:
            text_color = (0, 0, 255)
            
            if not violence_detected:
                violence_detected = True
                violence_start_frame = frame_count
                violence_start_time = time.time()
        else:
            violence_detected = False

        if violence_detected and frame_count == violence_start_frame + 30:
            # Capture the current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Send the alert with timestamp to the Telegram group
            message = f"Violence detected at {current_time}"
            with open('alert_frame.jpg', 'wb') as f:
                cv2.imwrite('alert_frame.jpg', frame * 255)
                bot.sendPhoto(telegram_group_id, open('alert_frame.jpg', 'rb'), caption=message)

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        # Write the frame with annotations to the output video
        out.write(output)

        cv2.imshow("Violence Detection", output)

        key = cv2.waitKey(1) & 0xFF
        frame_count += 1

        if key == ord("q"):
            break

    print("[INFO] Cleaning up...")
    vs.release()
    out.release()  # Release the output video writer
    cv2.destroyAllWindows()

input_video_file = 0
output_video_file = 'annotated_video.avi'  # Choose the output video format based on your preference
telegram_group_id = '-1002674875201'  # Replace with your Telegram group ID
save_annotated_video(input_video_file, output_video_file, telegram_group_id)
