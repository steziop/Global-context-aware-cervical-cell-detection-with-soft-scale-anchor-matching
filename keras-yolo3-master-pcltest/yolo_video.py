import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

def multi_detect_img(yolo):
    imagefile = input('Input imagefile path:')
    outdir = input('detection files path:')
    yolo.multi_detect_image(imagefile,outdir)
    
    yolo.close_session()

def classify_img(yolo):
    imagefile = input('Input imagefile path:')
    outfile = input('detection files path:')
    yolo.classify_image(imagefile,outfile)
    yolo.close_session()

def detect_with_classification(yolo):
    imagefile = input('Input imagefile path:')
    outdir = input('detection files path:')
    yolo.detect_with_classification(imagefile,outdir)
    yolo.close_session()

def detect_100(yolo):
    for i in range(100):
        imagefile = "/home/data/TCT_data100/testdata/00" + str(1500000 + i + 1) + "/TCT/slide" + str(i+1) + ".txt"
        outdir = "/home/data/TCT_data100/testdata/00" + str(1500000 + i + 1) + "/TCT/detections"
        if os.path.exists(outdir):
            continue
        else:
            os.mkdir(outdir)
            yolo.detect_with_classification(imagefile,outdir)
            
    yolo.close_session()
        



FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    parser.add_argument(
        '--multi_image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    parser.add_argument(
        '--scale', type=int,
        help='Number of predict layer to use, default ' + str(YOLO.get_defaults("scale"))
    )
    parser.add_argument(
        '--classify', default=False, action="store_true",
        help='classify image '
    )
    parser.add_argument(
        '--combine', default=False, action="store_true",
        help='detection results using classification as a filter'
    )
    parser.add_argument(
        '--detect100', default=False, action="store_true",
        help='detection results using classification as a filter on fucking 100 slides'
    )



    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    
    elif FLAGS.multi_image:
        print("Multi_Image detection mode")
        multi_detect_img(YOLO(**vars(FLAGS)))

    elif FLAGS.classify:
        print("Classify medical image mode")
        classify_img(YOLO(**vars(FLAGS)))

    elif FLAGS.combine:
        print("using classification results to filter detection results")
        detect_with_classification(YOLO(**vars(FLAGS)))

    elif FLAGS.detect100:
        print("using classification results to filter detection results on fucking 100 slides")
        detect_100(YOLO(**vars(FLAGS)))


    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
