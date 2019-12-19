# -e- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import glob
import tensorflow as tf

# ************* GPU memory Selection ***********************
# TensorFlow uses tf.ConfigProto() to configure the session
config = tf.ConfigProto() # A ProtocolMessage
config.gpu_options.allow_growth = True # .allow_growth: allocate/increase GPU memory based on demand-  GPU memory region needed by the TensorFlow process
#config.gpu_options.per_process_gpu_memory_fraction = 0.4 # determines the fraction (say 40%) of the overall amount of visible GPU memory is allocated
sess = tf.Session(config=config) # object encapsulates the environment 



class YOLO(object):
    _defaults = {
        "model_path": 'results/weights/uka_noTL_noAugfinal.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/classes_tools_sur.txt',
        "score" : 0.3,
        "iou" : 0.5,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()




    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        #print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,gt_label):
        start = timer()
        predicted_class = None
        score = 0

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        '''print(image_data.shape)'''
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        tickness_weight = 300 # to change the tickness of bbox
        thickness = (image.size[0] + image.size[1]) // tickness_weight 
        #thickness = (image.size[0] + image.size[1]) // 300

        #print('PRINT out_classes', out_classes) # out_classes is classe label
        #print(type(out_classes))

        
        # draw ground truth bounding box on the test images
        #gt_label = [312, 35, 432, 369]
        
        gt_label = gt_label.strip() # remove whitespace
        box = gt_label.split(' ')[1:] # gt_label is type of string
        #gt_class = gt_label.split(' ')[0]
        print( 'bbbox',box)
        
        for i in range(1):
            
            gt_class = 'gt' 

            top, left, bottom, right = int(box[1]), int(box[0]), int(box[3]), int(box[2]) # box should be list type
            
            draw_gt = ImageDraw.Draw(image)
            label_size = draw_gt.textsize(gt_class, font)
            
                        
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw_gt.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=(255, 255, 255))
            draw_gt.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=(0, 192, 192))
            draw_gt.text(text_origin, gt_class, fill=(0, 0, 0), font=font)
            del draw_gt


            
        # draw bounding box with predicted class score and label
        for i, c in reversed(list(enumerate(out_classes))): # out_classes is class label and  type is numpy arrary
            
            #print('PRINT OUT-classes', i, c)
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            ymin = round(top)
            xmin = round(left)
            ymax = round(bottom)
            xmax = round(right)

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
            

        end = timer()
        '''print('DETECTION TIME:', end - start) # detection time per image'''

        if len(out_boxes)==0:
            return image,predicted_class,score,0, 0, 0, 0
        else: 
            return image,predicted_class,score,xmin, ymin, xmax, ymax

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def detect_img(yolo):
    #test_files = glob.glob("test_data/*.jpg")
    frame_num = 0
    success_num = 0

    #2D_matrix = [[0 for i in range(32)] for j in range(32)]

    matrix = []

    for j in range(5):
        
        list = []
        for i in range(5):
            list.append(0)

        matrix.append(list)

    #print(matrix)


    gt_lists = []
    pd_lists = []
    path_files = []
    class_labels = []


   # Make a dictionary of all tools  
    with open("model_data/classes_tools_sur.txt") as f:
        tool_name = f.read().splitlines() # split line by line
     
    label_dict = dict([(tool_name, index) for index,tool_name in enumerate(tool_name)])
    label_dict['None']= 'None'     
     #label_dict = [dict(index,tool_name) for index,tool_name in enumerate(tools)]


   # read train label files and seperate file path and class label(ground truth)
    
    with open("results/annotation/val_uka_noTL_noAug.txt","r") as f: #val_annotation_path2.txt
        val_line = f.read().splitlines()
        for lines in val_line: 
            img_path = lines.split(' ')[0]
            
            '''Print ground truth image paths'''
            #print(img_path)

            path_files.append(img_path)
            class_label = lines.split(' ')[1].split(",")[4] 
            xmin = lines.split(' ')[1].split(",")[0]
            ymin = lines.split(' ')[1].split(",")[1]
            xmax = lines.split(' ')[1].split(",")[2]
            ymax = lines.split(' ')[1].split(",")[3]
            class_labels.append(class_label)
            #print(class_label) # string typei
            gt_list = "{} {} {} {} {}\n".format(class_label,xmin,ymin,xmax,ymax)
            gt_lists.append(gt_list)
    
    with open("results/gt_m2c_1.txt","w") as f:
        for i in gt_lists:
            f.write(i)
    

    for i,path in enumerate(path_files):
        frame_num += 1
        threshold = 0.5

        img_path = path
        gt_bbox = gt_lists[i]       

               
        print(img_path)
        print(gt_bbox)
        
        image = Image.open(img_path)
        r_image,predict_class,score, xmin, ymin, xmax, ymax = yolo.detect_image(image, gt_bbox)
        



        import cv2
        output_img_name = str(path).split(".jpg")
        output_img_name = str(output_img_name).split("/")[-1]
        output_img_name = str(output_img_name).replace(",","").replace(" ","").replace("'","").replace("]","").replace("[","")
        #print(output_img_name)
        save_path = 'results/output_test_image/'+ output_img_name + '_test.jpg'
        
        #print(save_path) 
                 
        cv2.imwrite(save_path, np.asarray(r_image)[..., ::-1])
        #cv2.imwrite("out.jpg", np.asarray(r_image)[..., ::-1])
        r_image.show()
                
        
        predict_label = label_dict[str(predict_class)]
        class_label = class_labels[i]
        class_label = str(class_label.rstrip("\n"))

        #print('this is predict label: ' + str(predict_label) + "predict label of shape" + str(type(predict_label))) 
        #print('this is class label: ' + str(class_label) + "class_label of shape " + str(type(class_label)))

        score = round(score,3)  # round off class score
        #print(score)

        pd_list = "{} {} {} {} {} {}\n".format(predict_class, score, xmin, ymin, xmax, ymax)
          
        pd_lists.append(pd_list)


        #print("class label: {} class lael type: {}".format(class_label,type(class_label))) 
        #print("predict label: {} predict lael type: {}".format(predict_label,type(predict_label)))


        if class_label == str(predict_label):
            true_label = int(class_label)
            pred_label = int(predict_label)
            #matrix[true_label][pred_label] +=1

            print("Detection Success:"+ " pred:" + str(predict_label) + " GT:"+ str(class_label))

        else: 
            print("Detection Failure:"+ " pred:" + str(predict_label) + " GT:"+ str(class_label))
            
    #print(len(matrix[0]))
    #print(matrix)
        
    #with open("./results/cm_m2c_1.txt", "w") as f:
    #    for i in range(5)
    #        f.write(str(matrix[i]))

    
    with open("./results/pd_m2c_1.txt","w") as f:
        for i in pd_lists:
            f.write(i)
   
   #print("class " + str(i) +": " + str(accuracy))
    

    yolo.close_session()

if __name__ == '__main__':
    
    #video_path = "video_iwc5.mp4"
    #detect_video(YOLO(), video_path)
    
    #img_path = "multiple.jpg"
    detect_img(YOLO())

