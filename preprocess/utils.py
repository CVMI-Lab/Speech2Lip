import cv2
import torch
import numpy as np
from PIL import Image
from face_parsing.BiSeNet import BiSeNet
import torchvision.transforms as transforms

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path=None,
                     img_size=(512, 512)):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + np.array([255, 255, 255])  # + 255

    num_of_class = np.max(vis_parsing_anno)
    # print(num_of_class)
    for pi in range(1, 14):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])

    for pi in range(14, 16):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 255, 0])
    for pi in range(16, 17):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 0, 255])
    for pi in range(17, num_of_class+1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    index = np.where(vis_parsing_anno == num_of_class-1)
    vis_im = cv2.resize(vis_parsing_anno_color, img_size,
                        interpolation=cv2.INTER_NEAREST)
    if save_im:
        cv2.imwrite(save_path, vis_im)

def face_parsing(output_path, image_path, model_path):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    img = Image.open(image_path)
    ori_size = img.size
    image = img.resize((512, 512), Image.BILINEAR)
    image = image.convert("RGB")
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    img = img.cuda()

    with torch.no_grad():
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    vis_parsing_maps(image, parsing, stride=1, save_im=True,
                        save_path=output_path, img_size=ori_size)
