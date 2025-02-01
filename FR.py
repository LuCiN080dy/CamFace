import cv2
import face_recognition
import numpy as np
import os
import hashlib
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage


# Load known faces from the dataset
def load_known_faces(dataset_path):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(dataset_path, file_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                encoding = encodings[0]
                name = os.path.splitext(file_name)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)
            else:
                print(f"No face encoding found in {file_name}")

    return known_face_encodings, known_face_names


# Generate a unique ID for a face
def generate_unique_id(face_encoding):
    hash_object = hashlib.sha256(face_encoding.tobytes())
    return hash_object.hexdigest()


# Send email notification for unknown visitors
def send_email_notification(unknown_face_image):
    admin_email = "XXX"
    sender_email = "YYY"
    sender_password = "ZZZ"

    subject = "Unknown Visitor Alert"
    body = "An unknown visitor was detected. Please check the attached image."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = admin_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = f"capture/unknown_{timestamp}.jpg"
    cv2.imwrite(image_path, unknown_face_image)
    with open(image_path, "rb") as f:
        attachment = MIMEImage(f.read())
        attachment.add_header("Content-Disposition", "attachment", filename=image_path)
        msg.attach(attachment)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, admin_email, msg.as_string())
        server.quit()
        print("Email sent to the administrator.")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Main function for the visitor counting and notification system
def visitor_system():
    dataset_path = r"E:\Face Detection and Recognition\datasets"
    known_face_encodings, known_face_names = load_known_faces(dataset_path)

    video_capture = cv2.VideoCapture(0)

    visitor_log = {}
    total_entered = 0
    total_exited = 0
    current_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        faces_seen_this_frame = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                unique_id = generate_unique_id(face_encoding)

                if unique_id not in visitor_log:
                    total_entered += 1
                    current_count += 1
                    entry_time = datetime.now()
                    visitor_log[unique_id] = {'name': name, 'entry_time': entry_time, 'exit_time': None}
            else:
                unique_id = generate_unique_id(face_encoding)
                if unique_id not in visitor_log:
                    total_entered += 1
                    current_count += 1
                    entry_time = datetime.now()
                    visitor_log[unique_id] = {'name': "Unknown", 'entry_time': entry_time, 'exit_time': None}
                    send_email_notification(frame)

            faces_seen_this_frame.append(unique_id)

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        for unique_id, log_entry in list(visitor_log.items()):
            if unique_id not in faces_seen_this_frame and log_entry['exit_time'] is None:
                total_exited += 1
                current_count -= 1
                log_entry['exit_time'] = datetime.now()

        frame_height, frame_width, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (frame_width, 50), (0, 0, 0), -1)
        text = f"Entered: {total_entered} | Exited: {total_exited} | Current: {current_count}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Visitor System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# Run the system
if __name__ == "__main__":
    visitor_system()