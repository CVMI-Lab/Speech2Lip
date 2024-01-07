import os
import sys
import face_alignment
import face_detection
from skimage import io
import numpy as np
from tqdm import tqdm

def detect_landmark():
    print('--- detect landmarks ---')
    ori_imgs_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print(f"Directory '{output_dir}' already exists.")

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')
    
    landmarks_dict = dict()
    for image_path in tqdm(sorted(os.listdir(ori_imgs_dir))):
        if image_path.endswith('.jpg'):
            input_file = io.imread(os.path.join(ori_imgs_dir, image_path))[:, :, :3]
            preds = fa.get_landmarks(input_file)
            
            if len(preds) > 0:
                lands = preds[0].reshape(-1, 2)[:,:2]
                np.savetxt(os.path.join(output_dir, image_path[:-3] + 'lms'), lands, '%f')
                landmarks_dict[image_path] = np.array(lands).astype(np.float32)

        if image_path not in landmarks_dict:
            print(image_path)
            
def detect_face_bbox():
    print('--- detect faces ---')
    ori_imgs_dir = sys.argv[1]
    output_dir = sys.argv[2]
        
    fa = face_detection.build_detector(
        "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3, device='cuda:0')

    face_bbox_dict = dict()
    for image_path in tqdm(os.listdir(ori_imgs_dir)):
        if '.jpg' in image_path:
            input_file = io.imread(os.path.join(ori_imgs_dir, image_path))[:, :, :3]
            preds = fa.detect(input_file.copy())
        
            if len(preds) > 1:
                print(image_path, len(preds))
                lands = preds[0]
                cur_face_bbox = np.array(lands).astype(np.float32)
                face_bbox_dict[image_path] = cur_face_bbox

            elif len(preds) > 0:
                lands = preds[0]
                cur_face_bbox = np.array(lands).astype(np.float32)
                face_bbox_dict[image_path] = cur_face_bbox

            if image_path not in face_bbox_dict:
                print(image_path)
            
    print('saving to ', os.path.join(output_dir.replace('landmarks', ''), 'face_bbox_dict.npy'))
    np.save(os.path.join(output_dir.replace('landmarks', ''), 'face_bbox_dict.npy'), face_bbox_dict)

if __name__=='__main__':
    detect_landmark()
    detect_face_bbox()