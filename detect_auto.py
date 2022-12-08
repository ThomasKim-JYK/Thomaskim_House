import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import sys
from utils.datasets import *
from tkinter import *
from tkinter import filedialog
import natsort

import os
from os import getcwd
from xml.etree import ElementTree as ET

class Detector():
    def __init__(self):
        self.objectList = []
        self.weights = r""  # 去官方下载v5 6.0的预训练权重模型
        self.dnn = False
        self.data = r""  # 选择你的配置文件(一般为.yaml)
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # 半精度化
        self.predefined_classes = []
        self.imgdir = r""  # 你需要标注的图片文件夹
        self.outdir = r""  # 你需要保存的xml文件夹
        self.detect_class = r""  # 你需要自动标注的类型
        self.root_window = None
        self.flag = False
        
    @torch.no_grad()
    def detect(self, image, save_img=False):
        source = image
        imgsz = opt.img_size
        # Directories
        # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    info_list = []
                    for *xyxy, conf, cls in reversed(det):
                        xyxy = torch.tensor(xyxy).view(-1).tolist()
                        info = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], int(cls)]
                        info_list.append(info)
                    return info_list
                else:
                    return None

    def create_annotation(self, xn):
        global annotation
        tree = ET.ElementTree()
        tree.parse(xn)
        annotation = tree.getroot()

    # Go through the whole .xml if already have the corresponding objects than skip
    def traverse_object(self, AnotPath):
        tree = ET.ElementTree(file=AnotPath)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        for Object in ObjectSet:
            ObjName = Object.find('name').text
            BndBox = Object.find('bndbox')
            x1 = int(BndBox.find('xmin').text)
            y1 = int(BndBox.find('ymin').text)
            x2 = int(BndBox.find('xmax').text)
            y2 = int(BndBox.find('ymax').text)
            self.objectList.append([x1, y1, x2, y2, ObjName])

    # Create ET subelments
    def create_object(self, root, objl):  # 参数依次，树根，xmin，ymin，xmax，ymax
        # 创建一级分支object
        _object = ET.SubElement(root, 'object')
        # 创建二级分支
        name = ET.SubElement(_object, 'name')
        # print(obj_name)
        name.text = str(objl[4])
        pose = ET.SubElement(_object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(_object, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(_object, 'difficult')
        difficult.text = '0'
        # 创建bndbox
        bndbox = ET.SubElement(_object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = '%s' % objl[0]
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = '%s' % objl[1]
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = '%s' % objl[2]
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = '%s' % objl[3]

    # Create .xml file
    def create_tree(self, image_name, h, w, imgdir):
        global annotation
        # 创建树根annotation
        annotation = ET.Element('annotation')
        # 创建一级分支folder
        folder = ET.SubElement(annotation, 'folder')
        # 添加folder标签内容
        folder.text = (imgdir)

        # 创建一级分支filename
        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_name

        # 创建一级分支path
        path = ET.SubElement(annotation, 'path')

        # path.text = getcwd() + '\{}\{}'.format(imgdir, image_name)  # 用于返回当前工作目录
        path.text ='{}/{}'.format(imgdir, image_name)  # 用于返回当前工作目录

        # 创建一级分支source
        source = ET.SubElement(annotation, 'source')
        # 创建source下的二级分支database
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'

        # 创建一级分支size
        size = ET.SubElement(annotation, 'size')
        # 创建size下的二级分支图像的宽、高及depth
        width = ET.SubElement(size, 'width')
        width.text = str(w)
        height = ET.SubElement(size, 'height')
        height.text = str(h)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'

        # 创建一级分支segmented
        segmented = ET.SubElement(annotation, 'segmented')
        segmented.text = '0'

    def pretty_xml(self, element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
        if element:  # 判断element是否有子元素
            if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
                element.text = newline + indent * (level + 1)
            else:
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
                # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
                # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
        temp = list(element)  # 将element转成list
        for subelement in temp:
            if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
                subelement.tail = newline + indent * (level + 1)
            else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
                subelement.tail = newline + indent * level
            self.pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作

    #Core part
    def work(self):
        with open(self.detect_class, "r") as f:  # 打开文件
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                self.predefined_classes.append(line)
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names
        IMAGES_LIST = os.listdir(self.imgdir)
        for image_name in natsort.natsorted(IMAGES_LIST):
            # print(image_name)
            # 判断后缀只处理图片文件
            if image_name.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                image = cv2.imread(os.path.join(self.imgdir, image_name))
                if image is None:
                    print(image_name+"图像为空请删除")
                    continue
                file_tail = os.path.splitext(image_name)[1]
                img_path = self.imgdir + '/' + str(image_name) 
                print(img_path)
                coordinates_list = self.detect(img_path)
                (h, w) = image.shape[:2]
                # xml_name = ('.\{}\{}.xml'.format(outdir, image_name.strip('.jpg')))
                xml_name = ('{}\{}.xml'.format(self.outdir, image_name.strip('.jpg')))
                if (os.path.exists(xml_name)):
                    self.create_annotation(xml_name)
                    self.traverse_object(xml_name)
                else:
                    self.create_tree(image_name, h, w, self.imgdir)
                if coordinates_list:
                    print(image_name + " is successfully labeled")
                    for coordinate in coordinates_list:
                        label_id = coordinate[4]
                        if (self.predefined_classes.count(names[label_id]) > 0):
                            object_information = [int(coordinate[0]), int(coordinate[1]), int(coordinate[2]),
                                                  int(coordinate[3]), names[label_id]]
                            if (self.objectList.count(object_information) == 0):
                                self.create_object(annotation, object_information)
                    self.objectList = []
                    # 将树模型写入xml文件
                    tree = ET.ElementTree(annotation)
                    root = tree.getroot()
                    self.pretty_xml(root, '\t', '\n')
                    # tree.write('.\{}\{}.xml'.format(outdir, image_name.strip('.jpg')), encoding='utf-8')
                    tree.write('{}\{}.xml'.format(self.outdir, image_name.strip(file_tail)), encoding='utf-8')
                else:
                    print(image_name)

    # Establish simple GUI
    def client(self):
        def creatWindow():
            self.root_window.destroy()
            window()

        def judge(str):
            if (str):
                text = "You have choosen" + str
            else:
                text = "No targets selected, please choose"
            return text

        def test01():
            self.imgdir = r""
            self.imgdir += filedialog.askdirectory()
            creatWindow()

        def test02():
            self.outdir = r""
            self.outdir += filedialog.askdirectory()
            creatWindow()

        def test03():
            self.data = r""
            self.data += filedialog.askopenfilename()
            creatWindow()

        def test04():
            self.weights = r""
            self.weights += filedialog.askopenfilename()
            creatWindow()

        def test05():
            self.detect_class = r""
            self.detect_class += filedialog.askopenfilename()
            creatWindow()

        def tes06():
            self.work()
            self.flag=True
            creatWindow()

        def window():
            self.root_window = Tk()
            self.root_window.title("")
            screenWidth = self.root_window.winfo_screenwidth()  # 获取显示区域的宽度
            screenHeight = self.root_window.winfo_screenheight()  # 获取显示区域的高度
            tk_width = 500  # 设定窗口宽度
            tk_height = 400  # 设定窗口高度
            tk_left = int((screenWidth - tk_width) / 2)
            tk_top = int((screenHeight - tk_width) / 2)
            self.root_window.geometry('%dx%d+%d+%d' % (tk_width, tk_height, tk_left, tk_top))
            self.root_window.minsize(tk_width, tk_height)  # 最小尺寸
            self.root_window.maxsize(tk_width, tk_height)  # 最大尺寸
            self.root_window.resizable(width=False, height=False)
            btn_1 = Button(self.root_window, text='Select unlabeled image file', command=test01,
                           height=0)
            btn_1.place(x=169, y=40, anchor='w')

            text = judge(self.imgdir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=70, anchor='w')
            btn_2 = Button(self.root_window, text='Select target annotation file', command=test02,
                           height=0)
            btn_2.place(x=169, y=100, anchor='w')
            text = judge(self.outdir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=130, anchor='w')
            btn_3 = Button(self.root_window, text='Select configuration file (.yaml)', command=test03,
                           height=0)
            btn_3.place(x=169, y=160, anchor='w')
            text = judge(self.data)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=190, anchor='w')

            # if(self.outdir and self.imgdir and self.data):
            btn_4 = Button(self.root_window, text='Select pretrained model (.pt)', command=test04,
                           height=0)
            btn_4.place(x=169, y=220, anchor='w')
            text = judge(self.weights)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=250, anchor='w')

            btn_5 = Button(self.root_window, text='Select autolabeled classes(.txt)', command=test05,
                           height=0)
            btn_5.place(x=169, y=280, anchor='w')
            text = judge(self.detect_class)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=310, anchor='w')

            btn_6 = Button(self.root_window, text='Lauch Automatic Labeling', command=tes06,
                           height=0)
            btn_6.place(x=169, y=340, anchor='w')
            if (self.flag):
                text = "Finished Labeling"
            else:
                text = "Waiting, in process"
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=370, anchor='w')
            self.root_window.mainloop()

        window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=384, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results',default=True)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    check_requirements(exclude=('pycocotools', 'thop'))

    detector = Detector()
    detector.client()
