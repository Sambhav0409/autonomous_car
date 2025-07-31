import numpy as np
from collections import deque
import cv2
from self_driving_car_pkg.Detection.Lanes.Lane_Detection import detect_lanes

class Control:
    def __init__(self, side='right'):
        self.side = side
        self.base_speed = 0.4  # Increased base speed
        self.max_speed = 1.2
        self.min_speed = 0.2
        self.speed = self.base_speed
        self.angle = 0.0
        self.angle_queue = deque(maxlen=5)
        self.lost_count = 0
        self.current_confidence = 0.0
        self.straight_frames = 0
        self.initialized = False
        
    def update(self, distance, curvature, confidence):
        self.current_confidence = confidence
        
        # Initial movement to get unstuck
        if not self.initialized:
            if confidence > 0.3:
                self.initialized = True
            else:
                self.lost_count += 1
                if self.lost_count > 10:  # After 10 frames of no detection
                    return self.base_speed * 0.6, 0.0  # Stronger initial push
                return 0.0, 0.0
        
        # Detect straight paths (more tolerant thresholds)
        is_straight = (abs(curvature) < 0.08 and 
                      abs(distance) < 0.2 and 
                      confidence > 0.5)
        
        if is_straight:
            self.straight_frames += 1
            # Ramp up speed on straights
            self.speed = min(self.max_speed, 
                           self.base_speed + (0.02 * self.straight_frames))
            self.angle = 0.0  # Force zero steering
            return self.speed, self.angle
        else:
            self.straight_frames = 0
        
        # Normal curved path operation
        steering = (distance * 1.5) + (curvature * 0.8)
        self.angle_queue.append(steering)
        self.angle = np.clip(np.mean(self.angle_queue), -0.8, 0.8)
        
        # Dynamic speed control
        if confidence > 0.7:
            self.speed = min(self.max_speed, self.speed + 0.01)
        else:
            self.speed = max(self.min_speed, self.speed - 0.02)
            
        return self.speed, self.angle

class Car:
    def __init__(self, side='right'):
        self.control = Control(side=side)
        self.side = side
        
    def driveCar(self, frame):
        try:
            if frame is None or frame.size == 0:
                return np.zeros((480,640,3), dtype=np.uint8), 0.0, 0.0
                
            # Process frame
            frame = cv2.resize(frame, (640, 480))
            distance, curvature, confidence, processed_img = detect_lanes(frame, self.side)
            
            # Get controls
            speed, angle = self.control.update(distance, curvature, confidence)
            
            # Display info
            mode = "STRAIGHT" if abs(angle) < 0.1 else "TURNING"
            mode_color = (0, 255, 0) if mode == "STRAIGHT" else (0, 165, 255)
            
            cv2.putText(processed_img, f"Speed: {speed:.2f}m/s", (20,40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(processed_img, f"Steering: {angle:.2f}", (20,80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            cv2.putText(processed_img, f"Confidence: {confidence:.2f}", (20,120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0,255,0) if confidence > 0.5 else (0,0,255), 2)
            cv2.putText(processed_img, f"Mode: {mode}", (20,160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            
            # Draw target path
            center_x = 320 + int(angle * 200)
            cv2.arrowedLine(processed_img, (320, 470), (center_x, 350), (0,200,255), 3)
            
            return processed_img, speed, angle
            
        except Exception as e:
            print(f"Drive error: {e}")
            return frame, 0.0, 0.0