import streamlit as st
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
from PIL import Image as pImage
import shutil
import os

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size,im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)])

class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))

class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length=60,transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        for i,frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top,right,bottom,left = faces[0]
                frame = frame[top:bottom,left:right,:]
            except:
                pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self,path):
        vidObj = cv2.VideoCapture(path) 
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    return image

def predict(model,img):
    fmap,logits = model(img.to('cuda'))
    logits = sm(logits)
    _,prediction = torch.max(logits,1)
    confidence = logits[:,int(prediction.item())].item()*100
    return [int(prediction.item()), confidence]

def save_uploaded_file(uploaded_file, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def run_model():

    page_bg_img = '''
    <style>
    body {
    background-image: url(https://github.com/karthiksagarN/DeepFake-Detection/blob/8b23d7fabf5e81db993ad71569626b3376bb6137/streamlit-webapp/background.jpeg);
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("DeepFake Video Detection")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "gif", "webm", "avi", "3gp", "wmv", "flv", "mkv"])

    if uploaded_file is not None:

        # Display the uploaded video
        st.video(uploaded_file)

        sequence_length = st.number_input("Enter sequence length", min_value=1, value=60)
        
        if sequence_length:
            model = Model(2)
            st.write("Model loaded successfully.")
            st.write("Starting prediction...")

            # Save the uploaded file to disk
            video_path = save_uploaded_file(uploaded_file, "uploaded_videos")

            video_dataset = validation_dataset([video_path], sequence_length=sequence_length, transform=train_transforms)

            # Load the trained model
            path_to_model = "/Users/karthiksagar/DeepFake-Detection/Trained-Models/best_model_accuracy.pt"
            # model.load_state_dict(torch.load(path_to_model))
            model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
            model.eval()
            prediction = predict(model, video_dataset[0])

            # Display prediction with icons or emojis
            prediction_text = "REAL" if prediction[0] == 1 else "FAKE"
            prediction_icon = "✅" if prediction[0] == 1 else "❌"
            st.write(f"Prediction: {prediction_text} {prediction_icon}")

            st.write("Confidence:", round(prediction[1], 2))

if __name__ == "__main__":
    run_model()