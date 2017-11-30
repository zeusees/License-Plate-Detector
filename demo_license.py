import numpy as np  
import sys,os  
import cv2
sys.path.append("/home/ly/workspace/caffe-master/python")
caffe_root = '/home/ly/workspace/caffe-master'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time; 

net_file= 'MobileNetSSD_test.prototxt'  
#caffe_model='license.caffemodel' 
caffe_model='lpd.caffemodel' 

test_dir = "license"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to genderate it.")
    exit()
#caffe.set_mode_cpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'license')


def preprocess(src):
    img = cv2.resize(src, (480,640))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img

    time_start=time.time()
    out = net.forward()  
    time_end=time.time()
    print time_end-time_start,  
    print "s" 

    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("SSD", origimg)
    cv2.imwrite('lena.png',origimg)
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
