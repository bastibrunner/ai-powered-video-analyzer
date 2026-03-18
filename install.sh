curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi4:latest
ollama pull partai/dorna-llama3:latest
ollama pull qwen:14b
apt install -y libgl1 
# Download YOLO11x model
yolo detect predict model=yolo11x.pt 
wget https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth?download=1 -O models/cnn14.pth
python -c "import whisper; whisper.load_model('large-v2')"