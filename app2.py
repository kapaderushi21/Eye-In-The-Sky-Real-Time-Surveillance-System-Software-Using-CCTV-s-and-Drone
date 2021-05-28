from Face import FaceCapture
from flask import Flask, render_template, Response, request
from camera import VideoCamera
from Detect import ObjectCapture

app=Flask(__name__)
video_camera = None
global_frame = None
video_cam = None
global_fram = None
vid_cam= None
glb_frame= None

@app.route("/",methods=['GET','Post'])
def example():
    return render_template('index.html')

@app.route("/motion",methods=['GET','Post'])
def video_stream1():
    global vid_cam
    global glb_frame

    if vid_cam == None:
        vid_cam = FaceCapture()
        
    while True:
        frame1 = vid_cam.face_recog()

        if frame1 != None:
            glb_frame = frame1
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + glb_frame + b'\r\n\r\n')

def video_stream2():
    global video_camera 
    global global_frame

    if video_camera == None:
        video_camera = ObjectCapture()
        
    while True:
        frame2 = video_camera.get_obj_frame()

        if frame2 != None:
            global_frame = frame2
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

def video_stream3():
    global video_cam
    global global_fram

    if video_cam == None:
        video_cam = VideoCamera()
        
    while True:
        frame3 = video_cam.get_frame()

        if frame3 != None:
            global_fram = frame3
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame3 + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_fram + b'\r\n\r\n')

@app.route('/video_viewer',methods=['GET','Post'])
def video_viewer():
    if request.method=='POST':
        if request.form['button'] == "Face Detection":
            return Response(video_stream1(), mimetype='multipart/x-mixed-replace; boundary=frame')
        if request.form['button'] == "Object Detection" :
            return Response(video_stream2(), mimetype='multipart/x-mixed-replace; boundary=frame')
        if request.form['button'] == "Motion Detection":
            return Response(video_stream3(), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return render_template('index.html')
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)