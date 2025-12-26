"""Check available cameras and test them."""
import cv2

print("Checking available cameras...\n")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera {i}: WORKING - Resolution {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"✗ Camera {i}: Opens but can't read frames")
        cap.release()
    else:
        print(f"✗ Camera {i}: Not available")

print("\nNow testing device 0 with window...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test", frame)
        print("✓ Frame displayed. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("✗ Can't read frame from device 0")
else:
    print("✗ Can't open device 0")
cap.release()
