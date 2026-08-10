#!/bin/bash
echo "Setting up Nyx..."

# Install system dependencies
sudo apt install git -y

# Setup venv and install packages
cd ~/ai-assistant
python3 -m venv venv
source venv/bin/activate
pip install requests faster-whisper webrtcvad-wheels audioop-lts python-dotenv google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client piper-tts

# Download piper voices
mkdir -p ~/piper-voices
cd ~/piper-voices
wget -nc https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -nc https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
wget -nc https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx
wget -nc https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx.json

# Add nyx alias
echo "alias nyx='cd /home/riza/ai-assistant && source venv/bin/activate && python nyx.py'" >> ~/.bashrc
source ~/.bashrc

echo "Done! Now create .env file with your API keys and copy credentials.json"
