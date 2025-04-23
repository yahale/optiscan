import requests
import json

api_endpoint_url = "http://localhost:8000/predict"

dry_eye_data = {
    'Gender': 'M',
    'Age': 30,
    'Sleep_duration': 7,
    'Sleep_quality': 8,
    'Stress_level': 3,
    'Sleep_disorder': 'N',
    'Wake_up_during_night': 'N',
    'Feel_sleepy_during_day': 'N',
    'Caffeine_consumption': 'Y',
    'Alcohol_consumption': 'N',
    'Smoking': 'N',
    'Smart_device_before_bed': 'Y',
    'Average_screen_time': 8,
    'Blue_light_filter': 'Y',
    'Discomfort_Eye_strain': 'Y',
    'Redness_in_eye': 'N',
    'Itchiness_Irritation_in_eye': 'N',
    'Systolic': 120,
    'Diastolic': 80,
}

image_file_path = "images-17.jpeg"  # Replace with the actual path to your image

try:
    print("Sending request...")
    print(f"Data being sent: {json.dumps(dry_eye_data, indent=4)}")

    files = {}
    if image_file_path:
        files = {"file": ("test_image.jpg", open(image_file_path, "rb"), "image/jpeg")}  # Open file *inside* the dict

    data = {"dry_eye_data": json.dumps(dry_eye_data)}
    response = requests.post(api_endpoint_url, files=files, data=data)

    print(response.request.headers)
    response.raise_for_status()
    prediction = response.json()
    print("API Response:")
    print(json.dumps(prediction, indent=4))

    if files:  #check if files is not empty
        files["file"][1].close() # Close the file after the request

except requests.exceptions.RequestException as e:
    print(f"Error sending request to API: {e}")
    if response is not None:
        print(f"API Error Response: {response.json()}")

except FileNotFoundError:
    print(f"Error: Image file not found at {image_file_path}")

finally:
    if 'image_file' in locals():
        image_file.close()