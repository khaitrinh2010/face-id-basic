import cv2

detector = cv2.CascadeClassifier("models/haarcascade_eye.xml")

img = cv2.imread("images/justin.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.1, #sau moi lan truot scale len ti le nay
    minNeighbors=20, # so tang phai vuot qua.
    minSize=(30, 30), # kick thuoc cua hinh vuong ban dau
)
print(faces)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
