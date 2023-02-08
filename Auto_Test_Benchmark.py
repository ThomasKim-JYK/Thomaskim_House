from detect_for_test import *
import torch

#overall accuracy
def measurement(a,b):
    num_correct = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            num_correct = num_correct + 1
    num_false = len(a) - num_correct
    print('The correct ratio is ', num_correct/len(a)) 


if __name__ == '__main__':
    #store the path of test_benchmark videos
    test_forder = ['./test_benchmark/Cube_CubeS_CylinderS/Cu_CuS_CyS_','./test_benchmark/Cube_CubeS_Triangle/Cu_CuS_T_',
                './test_benchmark/Cube_CylinderS_Triangle/Cu_CyS_T_','./test_benchmark/CubeS_Cylinder_TriangleS/CuS_Cy_TS_',
                './test_benchmark/Cylinder_CylinderS_TriangleS/Cy_CyS_TS_','./test_benchmark/Cylinder_TriangleS_Triangle/Cy_TS_T_']
    source_list = []
    for i in range(6):
        for j in range(1,4):
            element = test_forder[i] + str(j)
            source_list.append(element)
    #run 'detect' function on each videos 
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='C:/Users/Thomas/Desktop/yolov5_objects/runs/train/exp67/weights/best.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=384, help='inference size (pixels)') #384
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')#0.25
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS') #0.45
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    #store the probabilistic result; compare with standard solution
    result = []
    for i in range(18):
        source_name = source_list[i] + '.mp4'
        with torch.no_grad():
            list_no_order = detect(source_name,opt)
            list_no_order.sort()                       # rearrange the list from the smallest to biggest
            result.append(list_no_order)
    print("The detection result on test_benchmark is:")
    print(result)

    solution = [[0, 1, 3], [0, 1, 3], [0, 1, 3],        #Cube-0,CubeS-1,CylinderS-3 
                [0, 1, 4], [0, 1, 4], [0, 1, 4],        #Cube-0,CubeS-1,Triangle-4 
                [0, 3, 4], [0, 3, 4], [0, 3, 4],        #Cube-0,CylinderS-3,Triangle-4 
                [1, 2, 5], [1, 2, 5], [1, 2, 5],        #CubeS-1,Cylinder-2,TriangleS-5 
                [2, 3, 5], [2, 3, 5], [2, 3, 5],        #Cylinder-2,CylinderS-3,TriangleS-5 
                [2, 4, 5], [2, 4, 5], [2, 4, 5]]        #Cylinder-2,Triangle-4,TriangleS-5 
    
    measurement(result,solution)