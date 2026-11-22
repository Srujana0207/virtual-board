
import cv2
import time
import mediapipe as mp
import numpy as np

# Access MediaPipe Hand Landmarker API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

def angle_calc(m,n,p):

                A_x = m.x
                A_y = m.y
                B_x = n.x
                B_y = n.y
                C_x = p.x
                C_y = p.y

                BA_x = A_x - B_x
                BC_x = C_x - B_x
                BA_y = A_y - B_y
                BC_y = C_y - B_y

                dot_product = (BA_x*BC_x) + (BA_y*BC_y)

                BA_mag = np.sqrt(BA_x ** 2 + BA_y ** 2)
                BC_mag = np.sqrt(BC_x ** 2 + BC_y ** 2)

                if BA_mag!=0 and BC_mag!=0:
                    cosine = dot_product/(BA_mag * BC_mag)

                    cosine = np.clip(cosine,-1.0,1.0)

                    t=np.arccos(cosine)

                    tetha = np.degrees(t)

                    return tetha
                else:
                    return None

                    

#Configure the model
base_options = BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,  #image/video/live_cam
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
#Load the model
with HandLandmarker.create_from_options(options) as landmarker:

    #Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera could not be opened") #error

    else:
        start_time = time.time()

        x_positions=[]
        y_positions=[]

        canvas=None

        color=(255,0,0) #opencv uses BGR red means-(0,0,255)
        thickness = 5

        prev_point=None

        while True:
            #Capture a frame
            ret, frame = cap.read() #frame stores image as numpy array

            if not ret:
                print("Failed to capture frame") #error
                break
            frame = cv2.flip(frame, 1)
            #Convert BGR to RGB - mediapipe expects rgb input
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            #Convert NumPy array to MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, #rgb input
                data=rgb_frame
            )

            #Generate timestamp in milliseconds
            timestamp_ms = int((time.time() - start_time) * 1000) #convert into millisecs

            #Detect hands
            result = landmarker.detect_for_video(  #detection of landmarks on hand
                mp_image,
                timestamp_ms
            )

            #Display number of detected hands
            number_of_hands = len(result.hand_landmarks)

            if canvas is None:
                    frame_height,frame_width,channel=frame.shape
                    canvas = np.zeros((frame_height,frame_width,3),dtype=np.uint8)


            #result.hand_landmarks means counts hands

            '''print(frame.shape) -- (480,640,3) -- (height,width,channels) - channels - R,G,B - (3)
            '''
            frame_height,frame_width,channel = frame.shape
            for i in result.hand_landmarks:  #count hand
                for j in i:
                    pixel_x = j.x * frame_width #returns float
                    pixel_y = j.y * frame_height  #returns float
                    cv2.circle(frame,
                    (int(pixel_x),int(pixel_y)),  #convert both to integer as circle takes only integers
                    5,
                    (0,255,0),
                    3)

                    ''' syntax of -- cv2.circle(image,center,radius,color,thickness)'''
                    '''thickness = -1 means filled circle and 2 means circle outpline 2 pixels thick'''
                
                index_angle = angle_calc(i[5],i[6],i[7])
                middle_angle = angle_calc(i[9],i[10],i[11])

                pixel_x = int(i[8].x * frame_width)
                pixel_y = int(i[8].y * frame_height)

                x_positions.append(pixel_x)
                if (len(x_positions)>3):
                    x_positions.pop(0)
                
                y_positions.append(pixel_y)
                if (len(y_positions)>3):
                    y_positions.pop(0)

                avg_x = sum(x_positions)/len(x_positions)

                avg_y=sum(y_positions)/len(y_positions)

                current_point = (int(avg_x),int(avg_y))

                
                if index_angle>=160 and middle_angle<160:
                    #draw mode
                    print("DRAW MODE")
                    print(index_angle, middle_angle)
                    print("Drawing:", prev_point, current_point)
                    if prev_point is not None:
                        
                        cv2.line(canvas,prev_point,current_point,color,thickness)
                        print("Canvas pixel sum:", np.sum(canvas))

                    prev_point = current_point

                elif index_angle>=160 and middle_angle>=160:
                    #move

                    prev_point = None
                


                            
                #print(dir(mp.tasks.vision))
                #print(dir(mp.tasks.vision.HandLandmarksConnections))
                connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                #print(connections)
                for c in connections:
                    start = c.start
                    end = c.end
                    landmark_start = i[start]
                    landmark_end = i[end]

                    pixel_x1 = int(landmark_start.x * frame_width)
                    pixel_y1 = int(landmark_start.y * frame_height)
                    pixel_y2 = int(landmark_end.y * frame_height)
                    pixel_x2 = int(landmark_end.x * frame_width)

                    start_point = (pixel_x1,pixel_y1)
                    end_point = (pixel_x2,pixel_y2)

                    #cv2.line(image,startpoint,endpoint,color,thickness)

                    cv2.line(frame,
                    start_point,
                    end_point,
                    (0,255,0),
                    2)
                
                '''first_connection = next(iter(connections))

                print(first_connection)
                print(first_connection.__dict__)'''

            cv2.putText(
                frame, #numpy array of image
                f"Hands detected: {number_of_hands}",  #number of hands
                (20, 40),   #position
                cv2.FONT_HERSHEY_SIMPLEX,   #font
                1,  #font scale
                (0, 255, 0),    #color(rgb)
                2   #thickness
            )

            #use addWeighted to combine the camera captured frame and drawing canvas 
            #cv2.addWeighted(src1-first image,alpha-weight of firstimg,src2-second img,beta-wt of secod,gamma-scaler added to result)
            
            
            output = cv2.addWeighted(frame,1,canvas,1,0)
            cv2.imshow("AI Virtual Canvas - Hand Detection", output) 
            print("Canvas shape:", canvas.shape)
            print("Canvas size:", canvas.size)
            print("Canvas type:", type(canvas))
            print("Canvas sum:", np.sum(canvas))    #display on virtual canvas
            
            #Exit when q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):       #check the key which is pressed
                break

        #Release webcam
        cap.release()   #free up the camera
        cv2.destroyAllWindows()     #destroy all the memory


'''
for i in result.hand_landmarks:

    result.hand_landmarks = collection of detected hands.
i = all 21 landmarks belonging to ONE detected hand.
i[5] = landmark at index 5 (MCP).
i[6] = landmark at index 6 (PIP).
i[7] = landmark at index 7 (DIP).
'''
