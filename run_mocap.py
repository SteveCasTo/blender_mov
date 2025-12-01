import cv2
import time
from src.capture import PoseDetector
from src.solver import PoseSolver
from src.network import MocapSender
from src.exporter import MocapRecorder

def main():
    print("Starting Motion Capture System...")
    print("Press 'q' to quit.")
    print("Press 'r' to toggle recording.")
    
    detector = PoseDetector()
    solver = PoseSolver()
    sender = MocapSender(host='127.0.0.1', port=9000)
    recorder = MocapRecorder("output_animation.json")
    
    is_recording = False
    calibrated = False
    start_time = None
    
    print("Press 't' to start calibration (5 seconds countdown).")
    
    try:
        while True:
            image, results = detector.get_frame()
            
            if image is None:
                break
                
            # Check if we have pose landmarks
            has_pose = results.pose_landmarks is not None
                
            # Display Status
            if not calibrated:
                if start_time is None:
                    cv2.putText(image, "Press 't' to Calibrate", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    # Countdown Phase
                    elapsed = time.time() - start_time
                    remaining = 5.0 - elapsed
                    
                    if remaining > 0:
                        cv2.putText(image, f"Get Ready: {remaining:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(image, "STAND IN T-POSE", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        # Trigger Calibration
                        if has_pose:
                            print("Calibrating...")
                            solver.calibrate(results.pose_landmarks.landmark)
                            calibrated = True
                            print("Calibration Complete! Starting Stream.")
                        else:
                            cv2.putText(image, "No Body Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            # Reset if failed? Or keep trying? Let's keep trying.
                            start_time = time.time() - 5.1 # Force retry next frame
            
            if has_pose and calibrated:
                # Streaming Phase
                # 1. Solve Math (Landmarks -> Quaternions)
                # Pass the FULL results object (Pose + Hands)
                bone_data = solver.solve(results)
                
                # Debug: Print to verify data is being generated
                if "spine" in bone_data:
                    # Print hand positions for debugging
                    if "hand_ik.L" in bone_data:
                        # print(f"Hand L IK: {bone_data['hand_ik.L']}")
                        pass
                
                # 2. Send to Blender (Real-time)
                sender.send(bone_data)
                
                # 3. Record if enabled
                if is_recording:
                    recorder.record_frame(bone_data)
                    cv2.putText(image, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Motion Capture Feed', image)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                if not calibrated:
                    print("Starting Calibration Countdown...")
                    start_time = time.time()
            elif key == ord('r'):
                is_recording = not is_recording
                if is_recording:
                    recorder.start()
                    print("Recording started...")
                else:
                    recorder.save()
                    print("Recording saved.")
                    
    except KeyboardInterrupt:
        pass
    finally:
        detector.release()
        sender.close()
        if is_recording:
            recorder.save()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Make window resizable and larger
    cv2.namedWindow('Motion Capture Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Motion Capture Feed', 1280, 720)
    main()
