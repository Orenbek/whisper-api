from flask import Flask, jsonify, request
import numpy as np
import librosa
import whisper

app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/api/transcribe', methods=['POST'])
def upload():
    file = request.files['file']
    audio, sr = librosa.load(file)
    audio_array = np.array(audio)
    model = whisper.load_model("small")
    result = model.transcribe(audio_array)
    return jsonify(result)


if __name__ == '__main__':
    app.run()
