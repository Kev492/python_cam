import cv2
import os
import numpy as np 

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 나타냅니다.

ret, frame = cap.read()
# 이미지 저장 폴더 경로
save_folder = "C:/pyworkspace1/captured/finger_captured2"

# 폴더가 없다면 생성
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 손가락 검출을 위한 이미지 처리
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
ret, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 가장 큰 윤곽선을 찾아 손가락 영역 추출
max_area = 0
finger_frame = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        x, y, w, h = cv2.boundingRect(contour)
        finger_frame = frame[y:y+h, x:x+w]

if finger_frame is not None:
   # 손가락 영역의 중심 좌표 계산
    finger_center_x = w // 2
    finger_center_y = h // 2

    # 원하는 중앙 위치 계산 (예: 화면 중앙)
    screen_center_x = frame.shape[1] // 2
    screen_center_y = frame.shape[0] // 2

    # 중앙 위치로 이동하기 위한 변위 계산
    dx = screen_center_x - (x + finger_center_x)
    dy = screen_center_y - (y + finger_center_y)

    # 이동
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    finger_frame = cv2.warpAffine(finger_frame, M, (frame.shape[1], frame.shape[0]))

    # 손가락 영역을 파일로 저장
    image_count = len(os.listdir(save_folder))
    save_jpg = os.path.join(save_folder, f"captured_image_{image_count}.jpg")
    cv2.imwrite(save_jpg, finger_frame)

# 작업이 끝나면 웹캠을 닫습니다.
cap.release()
cv2.destroyAllWindows()