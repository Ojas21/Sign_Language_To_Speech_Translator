# ASL (American Sign Language) Recognition System

This project is designed to recognize and interpret American Sign Language (ASL) gestures in real-time using a trained deep learning model. It captures hand gestures from a webcam, processes them, and predicts the corresponding alphabet or word. Additionally, it provides options for audio output of the recognized gestures.

## Key Features

- **Real-time Gesture Recognition**: Captures hand gestures through a webcam and predicts the corresponding ASL alphabet or word.
- **Deep Learning Model**: Utilizes a pre-trained deep learning model for accurate gesture classification.
- **Textual and Audio Output**: Displays the recognized alphabets or words on the screen and optionally converts them into audio for better accessibility.
- **User Interaction**: Allows users to choose between recognizing single gestures or entire words.

## Dependencies

- **OpenCV**: Used for real-time video capture and image processing.
- **Keras with TensorFlow Backend**: Required for loading and using the pre-trained deep learning model.
- **NumPy**: Essential for numerical operations.
- **GTTS (Google Text-to-Speech)**: Enables conversion of recognized text into audio.
- **Win32com.client**: Provides access to Windows speech functionality for audio output.

## Usage

1. Run the `asl_recognition.py` script to start the ASL recognition system.
2. A live video stream will be displayed, showing your hand gestures.
3. The recognized alphabet or word will be displayed on the screen.
4. Optionally, choose between recognizing single gestures or entire words by entering `1` or `2` when prompted.
5. To exit the application, press `Esc`.

## Audio Output

- If you choose to recognize single gestures (`1`), each recognized alphabet will be spoken aloud as it appears on the screen.
- If you choose to recognize entire words (`2`), the entire recognized word will be spoken once it's completed.

## Note

- Ensure that the webcam is properly connected and configured before running the application.
- The accuracy of the recognition may vary depending on lighting conditions and background clutter.
- Adjust the `minValue` parameter in the script to optimize gesture recognition based on your environment.

## Author

This project is developed by [OJAS THENGADI].

