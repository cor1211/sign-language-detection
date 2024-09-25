import os
import cv2

# Tạo thư mục gốc chứa data
data_dir = './data-pre-handle-letter-ver2hands'
if not os.path.exists(data_dir):
   os.makedirs(data_dir)

# Số lượng phân loại
number_of_classes = 26

# Số lượng img cho 1 phân loại
data_size = 400

# Tạo đối tượng camera
cap = cv2.VideoCapture(1)

# Duyệt từng phân loại
for i in range(number_of_classes):
   if not os.path.exists(os.path.join(data_dir,str(i))):
      os.makedirs(os.path.join(data_dir,str(i)))
   
   print(f'Collecting data for class {i}...')
   

   while True:
      # Đọc frame từ camera, ret nhận true/false
      ret, frame = cap.read()
      if not ret:
         print('Không nhận được hình ảnh')
      else:
         # Thêm chữ vào khung frame
         cv2.putText(frame, 'Ready? Press Q to start!',(100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
         
         # Hiện thị frame
         cv2.imshow('Camera',cv2.flip(frame,1))
         # Set time nghỉ và key start
         if cv2.waitKey(1) == ord('q'):
            break
         
  
   # Biến số lượng ảnh
   quantity = 0
   while quantity < data_size:
      
      if quantity == 200:
         while True:
            if cv2.waitKey(1) == ord('q'):
               break
            
      # Đọc frame      
      ret, frame = cap.read()
      if not ret:
         print('Không nhận được hình ảnh')
      else:
         # Show frame 
         cv2.imshow('Camera',cv2.flip(frame,1))
         cv2.waitKey(25)
         # Lưu ảnh
         
         cv2.imwrite(os.path.join(data_dir,f'{i}',f'{quantity}.png'),frame)
         print(f'Saved {os.path.join(data_dir,f'{i}',f'{quantity}.png')}')
         # Tăng biến đếm ảnh
         quantity+=1
   
   # # Chờ để chụp tay phải
   # while True:
   #    # Đọc frame từ camera, ret nhận true/false
   #    ret, frame = cap.read()
   #    if not ret:
   #       print('Không nhận được hình ảnh')
   #    else:
   #       # Thêm chữ vào khung frame
   #       cv2.putText(frame, 'Ready? Press Q to start!',(100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
   #                  cv2.LINE_AA)
         
   #       # Hiện thị frame
   #       cv2.imshow('Camera',frame)
   #       # Set time nghỉ và key start
   #       if cv2.waitKey(1) == ord('q'):
   #          break

cap.release()
cv2.destroyAllWindows()
         
         
         
      
   