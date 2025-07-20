# node1.py - Object Detection Publisher
# Detects "board_shield" using YOLOv8 and publishes center-X coordinates via ZMQ

import cv2
import zmq
import json
import time
import signal
import sys
from ultralytics import YOLO

class ObjectDetectionNode:
    def __init__(self):
        self.running = True
        self.setup_zmq()
        self.setup_model()
        self.setup_camera()
        signal.signal(signal.SIGINT, self.signal_handler)

    def setup_zmq(self):
        try:
            self.context = zmq.Context()
            self.publisher = self.context.socket(zmq.PUB)
            self.publisher.bind("tcp://0.0.0.0:5555")
            print("✅ ZMQ Publisher bound to tcp://0.0.0.0:5555")
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ ZMQ setup failed: {e}")
            sys.exit(1)

    def setup_model(self):
        try:
            self.model = YOLO("best.pt")
            print("✅ YOLOv8 model 'best.pt' loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("💡 Ensure 'best.pt' exists in the current directory")
            sys.exit(1)

    def setup_camera(self):
        try:
            self.cap = cv2.VideoCapture(1)  # เปลี่ยนจาก 1 เป็น 0 หากภาพเพี้ยน
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("✅ Camera initialized (640x480)")
        except Exception as e:
            print(f"❌ Camera setup failed: {e}")
            sys.exit(1)
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Shutting down gracefully...")
        self.running = False
    
    def detect_and_publish(self):
        """Main detection loop"""
        print("🔍 Starting detection loop (Press 'q' to quit)")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Failed to read frame from camera")
                break
                
            # Run YOLO detection
            try:
                results = self.model(frame, verbose=False)[0]
                
                board_shield_found = False
                
                # Process detections
                for box in results.boxes:
                    if box is None:
                        continue
                        
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Skip low confidence detections
                    if conf < 0.5:
                        continue
                        
                    class_name = self.model.names[cls_id]
                    
                    # Check if it's our target class
                    if class_name == "board_shield":
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calculate center X coordinate
                        center_x = (x1 + x2) // 2
                        
                        # Publish center X coordinate
                        payload = {"x": center_x}
                        message = json.dumps(payload)
                        self.publisher.send_multipart([b"board_shield", message.encode()])
                        
                        print(f"📤 Published: board_shield at x={center_x} (conf={conf:.2f})")
                        
                        # Draw bounding box and label
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw center point
                        cv2.circle(frame, (center_x, (y1 + y2) // 2), 5, (255, 0, 0), -1)
                        
                        # Label with confidence
                        label = f"board_shield {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        board_shield_found = True
                        break  # Only track first detection
                
                # Show frame
                cv2.imshow("YOLOv8 - Board Shield Detection", frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🔲 'q' key pressed - exiting...")
                    break
                    
            except Exception as e:
                print(f"⚠️ Detection error: {e}")
                continue
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'cap'):
                self.cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'publisher'):
                self.publisher.close()
            if hasattr(self, 'context'):
                self.context.term()
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

def main():
    """Main function"""
    detector = ObjectDetectionNode()
    
    try:
        detector.detect_and_publish()
    except Exception as e:
        print(f"❌ Runtime error: {e}")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()