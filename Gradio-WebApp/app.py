import gradio as gr
from torch import nn
from PIL import Image
import base64
from io import BytesIO
import torch
from torchvision import transforms, models
# from frame_based_prediction import Model, final_detetcion  # Adjust this import according to your script

# Model class remains unchanged
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)  # Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.dp(self.linear1(x))


# Model and transformation setup
im_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def final_detetcion(frame_base64):
    frame_data = base64.b64decode(frame_base64)
    frame = Image.open(BytesIO(frame_data))

    # frame = Image.open(frame_path)  # Load image using PIL
    frame = transform(frame).unsqueeze(0) # Apply transformations and add batch dimension
    
    model_path = "/content/drive/MyDrive/DeepFake/Saved_Models/best_model_ff_data.pt"

    model = Model(num_classes=2).cuda()
    model.load_state_dict(torch.load(model_path))  # Load the saved model

    model.eval()
    with torch.no_grad():
        frame_tensor = frame.cuda() # Move the frame tensor to CUDA
        logits = model(frame_tensor)  # Assuming CUDA is available
        sm = nn.Softmax(dim=1)
        probabilities = sm(logits)
        confidence, prediction = torch.max(probabilities, dim=1)    

    if prediction.item() == 1:
        # print(f"Prediction: REAL VIDEO with {confidence * 100:.2f}% confidence")
        return "real", confidence.item() * 100
    else:
        # print(f"Prediction: DEEP-FAKE VIDEO with {confidence * 100:.2f}% confidence")
        return "fake", confidence.item() * 100
    
def process_frame(frame):
    # Convert the frame to PIL image
    pil_image = Image.fromarray(frame)
    # Convert PIL image to base64 string for model processing
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    result, confidence = final_detetcion(img_str)
    return result, confidence

def main(frame):
    result, confidence = process_frame(frame)
    return f"Prediction: {result} with {confidence:.2f}% confidence"

# Create Gradio interface for video input
interface = gr.Interface(
    fn=main,
    inputs=gr.Video(sources='webcam'),
    outputs=gr.Text(),
    live=True
)

if __name__ == "__main__":
    interface.launch()
