import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from CNN_v2 import ClassificationModel

def load_model(model_path):
    # Carrega o modelo completo, incluindo sua arquitetura
    model = ClassificationModel(num_classes=8) 
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Coloca o modelo em modo de avaliação
    return model

def prepare_transform():
    # Define as transformações que foram usadas durante o treinamento do modelo
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def classify_frame(frame, model, transform):
    # Converte o frame do OpenCV para uma imagem PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Aplica as transformações
    input_tensor = transform(image).unsqueeze(0)  # Adiciona uma dimensão de batch
    # Classifica o frame
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        confidence = max_prob.item() * 100  # Convertendo para percentual
    return predicted.item(), confidence

def load_class_names(txt_file_path):
    with open(txt_file_path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def process_video(video_path, model, transform):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Classifica o frame atual
        prediction, confidence = classify_frame(frame, model, transform)

        predicted_class_name = class_names[prediction]
        text = f'Classificacao: {predicted_class_name} ({confidence:.2f}%)'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = r'D:\Developements\INTELIGENCIA_ARTIFICIAL\Chronos_AI\Arquivos durante desenvolvimento\RedeNeuralWEG\pytorch\modelv2_script.pt'
    video_path = r'D:\Developements\INTELIGENCIA_ARTIFICIAL\Chronos_AI\Versao funcional\videos\VID_20240306_132820.mp4'

    txt_file_path = r'D:\Developements\INTELIGENCIA_ARTIFICIAL\Chronos_AI\Arquivos durante desenvolvimento\RedeNeuralWEG\pytorch\label.txt'
    class_names = load_class_names(txt_file_path)
    
    model = load_model(model_path)
    transform = prepare_transform()
    process_video(video_path, model, transform)
