import cv2
import os
import mediapipe
import pickle

# Tạo đối tượng làm việc với tay
mp_hands = mediapipe.solutions.hands
# Tạo đối tượng tạo landmark
mp_drawing = mediapipe.solutions.drawing_utils
# Tạo mô hình phát hiện tay
hands = mp_hands.Hands(
   static_image_mode = True,
   max_num_hands = 2,
   min_detection_confidence = 0.8,
   min_tracking_confidence = 0.2
)
# Tạo đối tượng camera
cap = cv2.VideoCapture(1)

data_dir = './data-pre-handle-letter-ver2hands'
features = []
labels = []
# label_maping = {'0':'A','1':'B','2':'D'}
for classify in os.listdir(data_dir):
   print(f'Đang xử lý trong folder {classify}')
   for img_file in os.listdir(os.path.join(data_dir,classify)):
      print(img_file)
      print(f'Đang xử lý ảnh {os.path.join(data_dir,classify,img_file)}')
      landmarks=[]
      # Đọc img
      image = cv2.imread(os.path.join(data_dir,classify,img_file))
      # Chuyển ảnh BGR sang RGB
      image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
      # Dò tìm tay trong ảnh
      results = hands.process(image_rgb)
      
      if results.multi_hand_landmarks:
         for hand_landmarks in results.multi_hand_landmarks:
            x_=[]
            y_=[]
            for i in range(len(hand_landmarks.landmark)):
               x = hand_landmarks.landmark[i].x
               y = hand_landmarks.landmark[i].y
               x_.append(x)
               y_.append(y)
               
            x_ = [ele - min(x_) for ele in x_]
            y_ = [ele - min(y_) for ele in y_]
            
            landmarks.extend(x_)
            landmarks.extend(y_)
            
      if len(landmarks) < 21*2:
         landmarks.extend([0] * (21*2-len(landmarks)))
            
      features.append(landmarks)
      labels.append(classify)

print(features)
for f in features:
   print(len(f))
print(labels)
print(len(features))
print(len(labels))

with open ('data-normalization-letter-2hands.pkl','wb') as f:
   pickle.dump((features,labels),f)
         
               
               
         
                         