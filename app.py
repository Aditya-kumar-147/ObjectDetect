import streamlit as st
import numpy as np
import cv2 as cv
net = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names','r') as f:
    classes = f.read().splitlines()
def main():
    st.title("YOLO Model")
    html_temp = """
    <div style="background-color:teal ; padding:10px">
    <h2 style="color:white; text-align:center;">Object Detection</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    if st.button('Start'):#button name is Classify
        cap = cv.VideoCapture(0)
        while True:
            ret, img = cap.read()
            if not ret:
                break
            height, width, ret = img.shape
            blob = cv.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop = False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)
            boxes = []
            confidences = []
            class_ids = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x,y,w,h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)
            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0,255,size = (len(boxes), 3))
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    cv.putText(img, label + " "+ confidence, (x, y+20), font, 2, (0,0,255), 2)
            cv.imshow("Image",img)
            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
              
if __name__=='__main__':
    main()
