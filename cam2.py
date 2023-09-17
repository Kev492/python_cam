import cv2
import os
import numpy as np 

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 나타냅니다.

# 이미지 저장 폴더 경로
save_folder = "C:/pyworkspace1/captured/finger_captured2"

# Create the save folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

while True:
    ret, frame = cap.read()

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
        # 검은 배경 이미지 생성
        black_background = np.zeros_like(frame)

        # 손가락 영역을 제외한 부분을 검은색으로 채우기 위한 마스크 생성
        mask = np.ones_like(frame) * 255
        mask[y:y+h, x:x+w] = 0

        # 손가락 영역을 제외한 배경 생성
        result_frame = np.where(mask == 0, finger_frame, black_background)

        # 손가락 영역을 파일로 저장 (파일명은 파일 수로 결정)
        image_count = len(os.listdir(save_folder))
        save_jpg = os.path.join(save_folder, f"captured_image_{image_count}.jpg")
        cv2.imwrite(save_jpg, result_frame)
        break
    
# 작업이 끝나면 웹캠을 닫습니다.
cap.release()
cv2.destroyAllWindows()