import cv2
import mediapipe
import os

# Tạo đối tượng làm việc với tay
mp_hands = mediapipe.solutions.hands

#Tạo đối tượng tạo và vẽ landmark
mp_drawing = mediapipe.solutions.drawing_utils

#Tạo mô hình phát hiện bàn tay
hands = mp_hands.Hands(
   static_image_mode =False, # False thì mô hình sẽ hoạt động trên video liên tục
   max_num_hands=2,  #Tối đa số bàn tay đc phát hiện trên 1 khung hình
   min_detection_confidence = 0.8, # Ngưỡng tin cậy để phát hiện
   min_tracking_confidence = 0.2, # Ngưỡng tin cậy để theo dõi
)

# Tạo đối tượng camera
cap = cv2.VideoCapture(1)

# Kiểm tra cam hoạt động
if cap.isOpened():
   while True:
      # Lấy thông tin về camera
      ret, frame = cap.read()
      if not ret:
         print('Không nhận đc frame')
         break
      # Chuyển đổi frame từ BGR sang RGB (vì cap.read() trả về BGR, mp làm vc với RGB)
      rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      
      # Xử lý frame hiện tại và phát hiện tay
      results = hands.process(rgb_frame)
      
      # Nếu phát hiện tay
      if results.multi_hand_landmarks:
         for one_hand in results.multi_hand_landmarks:
            # Tạo các landmark và vẽ
            mp_drawing.draw_landmarks(
                frame, 
                one_hand,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
      
      cv2.imshow('Landmark',cv2.flip(frame,1))
      if (cv2.waitKey(1) == ord('q')):
         break
      
cap.release()
cv2.destroyAllWindows()   
   
   