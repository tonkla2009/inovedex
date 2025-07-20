# node2.py - Servo Control Subscriber
# Receives board_shield coordinates via ZMQ and controls MG90 servo on Arduino

import zmq
import json
import time
import signal
import sys
from pyfirmata2 import Arduino

class ServoControlNode:
    def __init__(self):
        self.running = True
        self.arduino = None
        self.servo_pin = None
        self.current_angle = 90  # Start at center position
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.setup_arduino()
        self.setup_zmq()
        
    def setup_arduino(self):
        """Initialize Arduino connection and servo pin"""
        try:
            print("🔌 Connecting to Arduino on COM15...")
            self.arduino = Arduino('COM6')
            time.sleep(2)  # Allow Arduino to initialize
            
            # Configure pin D9 as servo (0-180 degrees)
            self.servo_pin = self.arduino.get_pin('d:9:s')
            
            # Move to center position initially
            self.servo_pin.write(self.current_angle)
            print(f"✅ Arduino connected, servo initialized at {self.current_angle}°")
            
        except Exception as e:
            print(f"❌ Arduino connection failed: {e}")
            print("💡 Check:")
            print("   - Arduino is connected to COM15")
            print("   - Arduino IDE is closed")
            print("   - Correct COM port in Device Manager")
            sys.exit(1)
    
    def setup_zmq(self):
        """Initialize ZMQ subscriber"""
        try:
            self.context = zmq.Context()
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.connect("tcp://localhost:5555")
            
            # Subscribe to board_shield topic
            self.subscriber.setsockopt(zmq.SUBSCRIBE, b"board_shield")
            
            # Set timeout for non-blocking receive
            self.subscriber.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
            
            print("✅ ZMQ Subscriber connected to tcp://localhost:5555")
            print("🔍 Subscribed to 'board_shield' topic")
            
        except Exception as e:
            print(f"❌ ZMQ setup failed: {e}")
            sys.exit(1)
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Shutting down gracefully...")
        self.running = False
    
    def map_x_to_angle(self, x_coord):
        """Map X coordinate (0-640) to servo angle (0-180)"""
        # Linear mapping: 0px -> 0°, 640px -> 180°
        angle = int((x_coord / 640.0) * 180)
        
        # Clamp to valid servo range
        angle = max(0, min(180, angle))
        
        return angle
    
    def move_servo_smooth(self, target_angle):
        """Move servo smoothly to target angle"""
        if abs(target_angle - self.current_angle) > 2:  # Only move if significant change
            steps = max(1, abs(target_angle - self.current_angle) // 5)  # Smooth movement
            
            if target_angle > self.current_angle:
                step = 1
            else:
                step = -1
                
            # Move in small increments for smooth motion
            for _ in range(steps):
                if not self.running:
                    break
                    
                self.current_angle += step
                self.current_angle = max(0, min(180, self.current_angle))
                
                try:
                    self.servo_pin.write(self.current_angle)
                    time.sleep(0.02)  # Small delay for smooth movement
                except Exception as e:
                    print(f"⚠️ Servo write error: {e}")
                    break
            
            # Final position
            try:
                self.current_angle = target_angle
                self.servo_pin.write(self.current_angle)
            except Exception as e:
                print(f"⚠️ Final servo position error: {e}")
    
    def listen_and_control(self):
        """Main control loop"""
        print("🎯 Servo control active (Ctrl+C to quit)")
        print("📡 Waiting for board_shield coordinates...")
        
        last_message_time = time.time()
        timeout_duration = 3.0  # Return to center after 3 seconds of no data
        
        while self.running:
            try:
                # Try to receive message (non-blocking with timeout)
                topic, message = self.subscriber.recv_multipart()
                
                # Parse JSON payload
                try:
                    data = json.loads(message.decode())
                    x_coord = int(data['x'])
                    
                    # Map to servo angle
                    target_angle = self.map_x_to_angle(x_coord)
                    
                    print(f"📥 Received x={x_coord} -> Moving to {target_angle}°")
                    
                    # Move servo
                    self.move_servo_smooth(target_angle)
                    
                    last_message_time = time.time()
                    
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"⚠️ Message parsing error: {e}")
                    
            except zmq.Again:
                # Timeout - no message received
                current_time = time.time()
                
                # Return to center if no messages for timeout duration
                if current_time - last_message_time > timeout_duration:
                    if self.current_angle != 90:  # Only move if not already at center
                        print("⏰ No data timeout - returning to center (90°)")
                        self.move_servo_smooth(90)
                        last_message_time = current_time  # Reset timer
                
                continue
                
            except Exception as e:
                print(f"❌ Communication error: {e}")
                time.sleep(0.1)
                continue
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Return servo to center position
            if self.servo_pin and self.arduino:
                print("🏠 Returning servo to center position...")
                self.servo_pin.write(90)
                time.sleep(0.5)
            
            # Close connections
            if self.arduino:
                self.arduino.exit()
                
            if hasattr(self, 'subscriber'):
                self.subscriber.close()
                
            if hasattr(self, 'context'):
                self.context.term()
                
            print("🧹 Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

def main():
    """Main function"""
    servo_controller = ServoControlNode()
    
    try:
        servo_controller.listen_and_control()
    except Exception as e:
        print(f"❌ Runtime error: {e}")
    finally:
        servo_controller.cleanup()

if __name__ == "__main__":
    main()