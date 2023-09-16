import cv2

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 나타냅니다. 다른 카메라를 사용하려면 숫자를 변경하세요.

# 프레임 읽기
ret, frame = cap.read()

# 이미지 파일로 저장
save_jpg = "captured_image.jpg"
cv2.imwrite(save_jpg, frame)

# 작업이 끝나면 웹캠을 닫습니다.
cap.release()
cv2.destroyAllWindows()

# 이미지 파일 경로 출력
print(save_jpg)