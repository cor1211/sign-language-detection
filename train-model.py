import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy
import cv2
import mediapipe
import time

# Convert labels -> letter
letter_label = {
   '0':'A',
   '1':'B',
   '2':'C',
   '3':'D',
   '4':'E',
   '5':'F',
   '6':'G',
   '7':'H',
   '8':'I',
   '9':'K',
   '10':'L',
   '11':'M',
   '12':'N',
   '13':'O',
   '14':'P',
   '15':'Q',
   '16':'R',
   '17':'S',
   '18':'T',
   '19':'U',
   '20':'V',
   '21':'W',
   '22':'X',
   '23':'Y'
}

# Init object do with hand
mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils

# Init model detected hands
hands = mp_hands.Hands(
   static_image_mode=False,
   max_num_hands = 1,
   min_detection_confidence = 0.9,
   min_tracking_confidence = 0.2
)

# Get feature and label 
with open('data-normalization-letter-2hands.pkl','rb') as f:
   features , labels = pickle.load(f)
   
# print(len(features),len(features[0]),len(features[99]) ,sep='\n')
# print(labels)

# Split data between train and test
# Shuffle -> make order random
# Straify -> sure for rate of label in train and test equals orginal
x_train, x_test, y_train, y_test = train_test_split(numpy.array(features),numpy.array(labels),test_size=0.3, shuffle=True, stratify=numpy.array(labels))

# Create object model 
model = RandomForestClassifier()

# Train
model.fit(x_train,y_train)
# print(x_train, y_train)
# Test predict
y_pred = model.predict(x_test)

# Rate rating exactly
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)

# Live predict with camera

# init object do with camera
cap = cv2.VideoCapture(1)

pos_text=''
if not cap.isOpened():
   print('Camera cant open')
else:
   # In loop
   while True:
      # Read frame 
      # time.sleep(0.1)
      ret, frame = cap.read()
      if not ret:
         print('Cant get frame from camera')
         break
      
      else:
         # Convert bgr->rgb
         image_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
         # Detect hands in image
         results = hands.process(image_rgb)
         if results.multi_hand_landmarks:
            print('Find hand')
            landmarks=[]
            for hand_landmarks in results.multi_hand_landmarks:
               # Define landmark and draw in hands
               
               mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
               )
               
               x_=[]
               y_=[]
               
               # iterate over all landmarks in 1 hand by index
               for i in range(len(hand_landmarks.landmark)):
                  x = hand_landmarks.landmark[i].x 
                  y = hand_landmarks.landmark[i].y
                  # Add
                  x_.append(x)
                  y_.append(y)
               h,w,_ = frame.shape
               # print(h,w,_)
               pos_text=(int(min(x_)*w)-20,int(min(y_)*h)-40)
               cv2.rectangle(frame,(int(min(x_)*w)-20,int(min(y_)*h)-20),(int(max(x_)*w)+20,int(max(y_)*h)+20),(0,255,0),2)
               # Normalization data
               x_ = [ele - min(x_) for ele in x_]
               y_ = [ele - min(y_) for ele in y_]
               
               landmarks.extend(x_)
               landmarks.extend(y_)
            
            if (len(landmarks)< 21*2):
               landmarks.extend([0]*(21*2 - len(landmarks)))
            # print('landmarks',landmarks)
            y_pred = model.predict([numpy.array(landmarks)])
            
            print(y_pred)
            
            print(letter_label[y_pred[0]])
            cv2.putText(frame, letter_label[y_pred[0]],pos_text,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0 , 0), 2)
                  
      cv2.imshow('Landmark', cv2.flip(frame,1))
      # cv2.imshow('Landmarks', frame)
            
      if cv2.waitKey(25) == ord('q'):
         break
               
cap.release()
cv2.destroyAllWindows()