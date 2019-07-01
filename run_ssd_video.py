from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
from vision.utils import box_utils
import torch 
from vision.ssd.mobilenetv1_ssd_particle import create_mobilenetv1_ssd_particle, create_mobilenetv1_ssd_particle_predictor

if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path> <is_crop>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]
is_crop = False
if len(sys.argv) > 5:
    is_crop = True
print(is_crop)   
class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-p':
    net = create_mobilenetv1_ssd_particle(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=15000)
elif net_type == 'mb1-ssd-p':
    predictor = create_mobilenetv1_ssd_particle_predictor(net, candidate_size=15000)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

if is_crop:
    cap = cv2.VideoCapture(image_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    SIZE = 300
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (3300,2700))
    frame_idx = 0
    while(cap.isOpened()):
        frame_idx+=1
        print(frame_idx)
        ret, orig_image = cap.read()
        if ret == False:
            break
        # orig_image = cv2.imread(image_path)
        height, width,_ = orig_image.shape
        toppad = SIZE - height % SIZE
        leftpad = SIZE - width % SIZE 
        orig_border=cv2.copyMakeBorder(orig_image, top=toppad//2, bottom=toppad//2, 
        left=leftpad//2, right=leftpad//2, 
        borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
        height, width,_ = orig_border.shape
        border = cv2.cvtColor(orig_border, cv2.COLOR_BGR2GRAY)[:,:,None]

        pieces = []
        offsets = []
        #Patches orig
        for row in range(1,width//SIZE + 1):
            for col in range(1,height//SIZE + 1):
                pieces.append(border[(col-1)*SIZE:col*SIZE,(row-1)*SIZE:row*SIZE,:])
                offsets.append(((row-1)*SIZE,(col-1)*SIZE))

        boxes, labels, probs = predictor.predict_pieces(pieces, offsets, -1, 0.5)

        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(orig_border, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        out.write(orig_border)
        if frame_idx > 20:
            break
    cap.release()
    out.release()

