import cv2
import numpy as np

def detect_lanes(frame, side='right'):
    """Enhanced lane detection with straight path boosting"""
    # Initialize defaults
    distance = 0.0
    curvature = 0.0
    confidence = 0.0
    debug_frame = np.zeros((480, 640, 3), dtype=np.uint8) if frame is None else frame.copy()

    try:
        # Verify input
        if frame is None or frame.size == 0:
            return distance, curvature, confidence, debug_frame

        # Standardize frame size
        height, width = 480, 640
        frame = cv2.resize(frame, (width, height))
        debug_frame = frame.copy()
        
        # Convert to HLS color space
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        
        # Color thresholds (optimized for lane markings)
        yellow_lower = np.array([15, 50, 50])
        yellow_upper = np.array([35, 255, 255])
        white_lower = np.array([0, 150, 0])
        white_upper = np.array([180, 255, 255])
        
        # Create masks
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hls, white_lower, white_upper)
        
        # Region of interest (wider area)
        mask = np.zeros_like(yellow_mask)
        vertices = np.array([[
            (0, height),
            (width//3, height//2),
            (2*width//3, height//2), 
            (width, height)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, [vertices], 255)
        
        # Apply masks
        yellow_masked = cv2.bitwise_and(yellow_mask, mask)
        white_masked = cv2.bitwise_and(white_mask, mask)
        
        # Edge detection
        yellow_edges = cv2.Canny(yellow_masked, 50, 150)
        white_edges = cv2.Canny(white_masked, 50, 150)
        
        # Detect lines
        def get_lines(edges, min_len=30):
            lines = cv2.HoughLinesP(edges, 2, np.pi/180, 50,
                                  minLineLength=min_len, maxLineGap=100)
            return lines if lines is not None else []
        
        yellow_lines = get_lines(yellow_edges, 40)
        white_lines = get_lines(white_edges, 30)
        
        # Filter lines by slope
        def filter_lines(lines, min_slope, max_slope):
            filtered = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2-y1)/(x2-x1)
                if min_slope <= slope <= max_slope:
                    filtered.append(line[0])
                    cv2.line(debug_frame, (x1,y1), (x2,y2), (0,255,255), 2)
            return filtered
        
        left_yellow = filter_lines(yellow_lines, -np.inf, -0.2)
        right_yellow = filter_lines(yellow_lines, 0.2, np.inf)
        white_lane = filter_lines(white_lines, -0.5, 0.5)
        
        # Calculate position between lanes
        if side == 'right' and right_yellow and white_lane:
            right_avg = np.mean(right_yellow, axis=0)
            white_avg = np.mean(white_lane, axis=0)
            target_x = (right_avg[0] + white_avg[0]) / 2
            distance = (width//2 - target_x) / width
            
            # Calculate curvature
            dx = right_avg[2] - right_avg[0]
            if dx != 0:
                curvature = (right_avg[3] - right_avg[1]) / dx
            
            # Boost confidence for straight paths
            confidence = min(1.0, (len(right_yellow) + len(white_lane)) / 8.0)
            if abs(curvature) < 0.1:  # Nearly straight
                confidence = min(1.0, confidence * 1.3)  # Confidence boost
            
            cv2.line(debug_frame, (width//2, height), (int(target_x), height//2), (0,255,0), 3)
        
        elif side == 'left' and left_yellow and white_lane:
            left_avg = np.mean(left_yellow, axis=0)
            white_avg = np.mean(white_lane, axis=0)
            target_x = (left_avg[0] + white_avg[0]) / 2
            distance = (width//2 - target_x) / width
            
            dx = left_avg[2] - left_avg[0]
            if dx != 0:
                curvature = (left_avg[3] - left_avg[1]) / dx
            
            confidence = min(1.0, (len(left_yellow) + len(white_lane)) )/ 8.0
            if abs(curvature) < 0.1:
                confidence = min(1.0, confidence * 1.3)
            
            cv2.line(debug_frame, (width//2, height), (int(target_x), height//2), (0,255,0), 3)
        
        return distance, curvature, confidence, debug_frame
        
    except Exception as e:
        print(f"Detection error: {e}")
        return distance, curvature, confidence, debug_frame