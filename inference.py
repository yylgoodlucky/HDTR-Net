import os, cv2
import argparse
import face_alignment
import pickle, time
import shutil

from os.path import join
from tqdm import tqdm

import torch
import numpy as np 
from networks.models import Generator
from dataset.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader

border = 2

split_image_cmd = 'ffmpeg -hwaccel cuvid -hide_banner -loglevel quiet -y -i {} -r 25 {}/%05d.png'
split_wav_cmd = 'ffmpeg -hwaccel cuvid -hide_banner -loglevel quiet -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'
fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cuda:0')

def load_model(args):
    Model = Generator(args)
    print("Load checkpoint from: {}".format(args.model_path))
    checkpoint = torch.load(args.model_path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v

    Model.load_state_dict(new_s)
    Model = Model.cuda().eval()
    
    return Model

def pixel_connet(img, landmarks):
    # 将关键点连线
    for i in range(60, 67):
        cv2.line(img, landmarks[i].astype(int), landmarks[i+1].astype(int), 0, 1)
        
    # 连接首尾关键点，闭合形成多边形
    cv2.line(img, landmarks[67].astype(int), landmarks[60].astype(int), 0, 1)
    return img


def cutting_mouth(landmarks):
    landmarks = landmarks.astype(int)
    
    x_jaw, y_jaw = landmarks[8]    # 下巴
    x_nose, y_nose = landmarks[30]   # 鼻尖
    x_left_mouth, y_left_mouth = landmarks[48]  # 左嘴角
    x_right_mouth, y_right_mouth = landmarks[54]  # 右嘴角

    if (y_jaw - y_nose) > (x_right_mouth - x_left_mouth):
        padlen = ((y_jaw - y_nose) - (x_right_mouth - x_left_mouth))/2
        x_left_mouth -= padlen
        x_right_mouth += padlen
    elif (y_jaw - y_nose) < (x_left_mouth - x_right_mouth):
        padlen = ((x_right_mouth - x_left_mouth) - (y_jaw - y_nose))/2
        y_nose -= padlen
        y_jaw += padlen
        
    return 	int(x_left_mouth), int(y_nose), int(x_right_mouth), int(y_jaw)


def rotation(full_image, in_mask, in_mean_mask, image, landmarks):
    pt1 = landmarks[48]    # left_corner
    pt2 = landmarks[54]    # right_corner
    
    angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180 / np.pi     # 计算旋转角度
    center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)              # 计算中心点坐标
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)          # 旋转图像
        
    aligned_in_mask = cv2.warpAffine(in_mask, rotation_matrix, (image.shape[1], image.shape[0]))
    aligned_in_mean_mask = cv2.warpAffine(in_mean_mask, rotation_matrix, (image.shape[1], image.shape[0]))
    aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    aligned_full_image = cv2.warpAffine(full_image, rotation_matrix, (full_image.shape[1], full_image.shape[0]))

    keypoints = np.array(landmarks)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    keypoints_homogeneous = np.concatenate((keypoints, np.ones((keypoints.shape[0], 1))), axis=1)     # 将关键点坐标转换为齐次坐标
    keypoints_rotated = np.dot(rotation_matrix, keypoints_homogeneous.T).T     # 应用旋转矩阵进行坐标变换
    keypoints_rotated = keypoints_rotated[:, :2]                               # 提取旋转后的关键点坐标
    return aligned_full_image, aligned_in_mask, aligned_in_mean_mask, aligned_image, keypoints_rotated, rotation_matrix

def get_mask(image, landmarks):
    # cut_image = aligned_img[int(y_nose):int(y_jaw), int(x_left_mouth):int(x_right_mouth), :]
    # Difinate the mouth region landmarks indices
    mouth_indices = list(range(48, 60))  # mouth landmarks indices
    tooth_indices = list(range(60, 67))    # tooth landmarks indices

    # Create mouth region mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mouth_mask = np.zeros_like(gray)
    tooth_mask = np.zeros_like(gray)

    # Accroding to landmarks indices to build mouth region polygon
    mouth_ = np.array(landmarks[mouth_indices], dtype=np.int32)
    tooth_ = np.array(landmarks[tooth_indices], dtype=np.int32)
    cv2.fillPoly(mouth_mask, [mouth_], 255)
    cv2.fillPoly(tooth_mask, [tooth_], 255)
    
    # Set the mouth area to black
    mouth_area = cv2.bitwise_and(image, image, mask=mouth_mask)
    mouth_area[mouth_mask == 0] = 0

    tooth_area = cv2.bitwise_and(image, image, mask=tooth_mask)
    tooth_area[tooth_mask == 0] = 0

    lips_area = mouth_area - tooth_area
    lips_mask = mouth_mask - tooth_mask
    in_mask = image - mouth_area
    
    # mean_tooth = np.mean(image[np.where(tooth_mask == 255)])
    # mean_lip = np.mean(image[np.where(lips_mask == 255)])

    mean_tooth = np.mean(image[tooth_area != 0])
    mean_lip = np.mean(image[lips_area != 0])
    mean_mask = np.zeros_like(image)

    mean_mask[np.where(tooth_mask == 255)] = mean_tooth
    mean_mask[np.where(lips_mask == 255)] = mean_lip
    
    # mean_mask[tooth_mask != 0] = mean_tooth
    # mean_mask[lips_mask != 0] = mean_lip
    in_mean_mask = pixel_connet(mean_mask, landmarks)

    return in_mask, in_mean_mask, image

def prepare_data(full_frame_path, temp_dir):
    # Detect face landmarks 
    print(f'==> detect face landmarks ...')
    frame_list = sorted(os.listdir(full_frame_path))

    landmarks_dict = {}
    for index in tqdm(frame_list):
        # index_base = index.replace('.jpg', '')
        image = cv2.imread(join(full_frame_path, index))
        try:
            preds = fa_3d.get_landmarks(image)
        except Exception as e:
            print(f'Catched the following error: {e}')
            preds = None
            continue
        lmark = preds[0][:,:2]
        landmarks_dict[index] = lmark
    
    landmarks_path = join(temp_dir, 'landmarks')
    os.makedirs(landmarks_path, exist_ok=True)
    
    if os.path.exists(join(landmarks_path, 'landmarks.pkl')):
        os.remove(join(landmarks_path, 'landmarks.pkl'))

    with open(join(landmarks_path, 'landmarks.pkl'), 'wb') as f:
        pickle.dump(landmarks_dict, f)

    # Crap based on landmarks 
    align_crop_image_path = join(temp_dir, 'align_crop_image')
    os.makedirs(align_crop_image_path, exist_ok=True)
    
    print(f'==> face align ...')
    with open(join(landmarks_path, 'landmarks.pkl'), 'rb') as f:
        read_landmarks = pickle.load(f)
    
    image_config = {}
    image_coor = {}
    image_hw = {}
    image_config_path = join(temp_dir, 'image_config')
    os.makedirs(image_config_path, exist_ok=True)
    if os.path.exists(join(image_config_path, 'image_config.pkl')):
        os.remove(join(image_config_path, 'image_config.pkl'))

    with open(join(image_config_path, 'image_config.pkl'), 'wb') as config:
        for i in tqdm(frame_list):
            # pdb.set_trace()
            landmark = read_landmarks[i]
            
            image_path = join(full_frame_path, i)
            image = cv2.imread(image_path)
            
            in_mask, in_mean_mask, cut_image = get_mask(image, landmark)
            
            x0, y0, x1, y1 = cutting_mouth(landmark)
            image_coor[i] = (x0, y0, x1, y1)
            
            h, w, _ = image.shape
            if x0 < 0: x0 = 0
            if y0 < 0: y0 = 0
            if x1 > w: x1 = w - 1
            if y1 > h: y1 = h - 1
            
            base_image = i.replace('.png', '')
            cv2.imwrite(join(align_crop_image_path, base_image + '_mask.png'), in_mask[y0:y1, x0:x1, :])
            cv2.imwrite(join(align_crop_image_path, base_image + '_mean_mask.png'), in_mean_mask[y0:y1, x0:x1, :])
            cv2.imwrite(join(align_crop_image_path, base_image + '_image.png'), cut_image[y0:y1, x0:x1, :])
            h, w, _ = cut_image[y0:y1, x0:x1, :].shape
            image_hw[i] = (h, w)
        image_config['image_hw'] = image_hw
        image_config['image_coor'] = image_coor
        pickle.dump(image_config, config)
    return frame_list


def save_results(SR_images, path, index):
    save_path = join(path, 'model_out')
    os.makedirs(save_path, exist_ok=True)
    
    for ind in range(SR_images.shape[0]):
        pred_hq = SR_images[ind, :].numpy() * 255.
        pred_hq = np.transpose(pred_hq.astype(int), (1, 2, 0))
    
        cv2.imwrite(join(save_path, '{}.png'.format(str(index+ind).zfill(5))), pred_hq)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default='./sample/00001.mp4', help='Input video to clear mouth region')
    parser.add_argument('--temp_dir', type=str, default='./test_result', help='Temp directory to save output')
    parser.add_argument('--model_path', type=str, default='./checkpoint/lip-clarity-model.pth', 
                        help='Root path of pretrained SR model')
    
    parser.add_argument("--mask_channel", type=int, default=6, help='Channel of the mouth mask and mouth contour')
    parser.add_argument("--ref_channel", type=int, default=3, help='Channel of the reference channel')
    parser.add_argument("--mouth_size", type=int, default=96, help='Size of the crop mouth region image')
    parser.add_argument("--test_batch_size", type=int, default=8, help='Size of the test batch')
    parser.add_argument("--test_workers", type=int, default=0, help='Number of workers to run the test')
    
    args = parser.parse_args()
    
    # --- Split image from video --- #
    video_basename = os.path.basename(args.input_video).replace('.mp4', '')
    result_path = join(args.temp_dir, video_basename)
    
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    
    os.makedirs(result_path, exist_ok=True)    
    full_frame_path = join(result_path, 'full_frame')
    wav_path = join(result_path, 'audio.wav')
    os.makedirs(full_frame_path, exist_ok=True)
    os.system(split_image_cmd.format(args.input_video, full_frame_path))
    os.system(split_wav_cmd.format(args.input_video, wav_path))

    # --- Prepare our input data --- #
    print(f'[info] Step1: Prepare input data ...')
    frame_list = prepare_data(full_frame_path, result_path)
    
    align_crop_image_path = join(result_path, 'align_crop_image')
    
    # --- Load model and inferece --- #
    print('[info] Step2: load model ...')
    HQ_Generator = load_model(args)

    dataset = InferenceDataset(frame_list, align_crop_image_path, args.mouth_size)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=int(args.test_workers),
                            drop_last=False)
    
    start = time.time()

    print(f"[info] Step3: save results ...")
    start_index = 1
    for mouth_mask, mouth_contour, referece in dataloader:
        print(f'[info]: Processing cropped image to clear mouth region : {start_index}-{start_index+args.test_batch_size}')
        
        with torch.no_grad():
            pred_hq = HQ_Generator(mouth_mask, mouth_contour, referece).cpu()
            
        save_results(pred_hq, result_path, index=start_index)
        start_index += args.test_batch_size
 
    end = time.time()
    # print(end - start)
    
    save_path = join(result_path, 'model_out')
    synthesis_CMD = "ffmpeg -y -loglevel warning " + \
            "-thread_queue_size 8192 -i {} " + \
            "-thread_queue_size 8192 -i {}/%05d.png " + \
            "-vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest {}/{}_{}.mp4"
    merged_CMD = "ffmpeg -hide_banner -y -loglevel warning " + \
            "-thread_queue_size 8192 -i {}/%05d.png " + \
            "-thread_queue_size 8192 -i {}/%05d.png " + \
            "-i {} " + \
            "-filter_complex hstack=inputs=2 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p {}/merge_{}.mp4"
    
    os.system(synthesis_CMD.format(wav_path, save_path, result_path, "out", video_basename))
    
    
    # --- InverAffineTransform --- # 
    """
    info: fixed not rotated(no warpaffine) by default
    """
    print('[info] Step4: Inver Affine Transform (Being here means wearing paste back) ...')
    warpAffine_parm_path = join(result_path, 'warpAffine_param')
    
    image_config_path = join(result_path, 'image_config')
    with open(join(image_config_path, 'image_config.pkl'), 'rb') as f:
        image_config = pickle.load(f)
    print('==> paste background ...')
    for index in tqdm(frame_list):
        full_image = cv2.imread(join(full_frame_path, index))

        # paste bask
        h, w = image_config['image_hw'][index]
        out_image = cv2.imread(join(save_path, index))
        out_image = cv2.resize(out_image, (h, w))
        
        x0, y0, x1, y1 = image_config['image_coor'][index]
        
        # create mouth mask
        mask = np.zeros(full_image.shape[:2], dtype=np.uint8)
        souce_img = full_image.copy()
        souce_img[y0:y0+h, x0:x0+w, :] = out_image
        mask[y0:y0+h, x0:x0+w] = 255

        # seamless
        mixed_image = cv2.seamlessClone(souce_img, full_image, mask, (x0+w//2, y0+h//2), cv2.NORMAL_CLONE)

        # full_image[y0:y1, x0:x1, :] = out_image
        finout_image_path = join(result_path, 'fin_out')
        os.makedirs(finout_image_path, exist_ok=True)
        cv2.imwrite(join(finout_image_path, index), mixed_image)
        
    os.system(synthesis_CMD.format(wav_path, finout_image_path, result_path, "result", video_basename))
    os.system(merged_CMD.format(full_frame_path, finout_image_path, wav_path, result_path, video_basename))
    print(f"[info]: Result videos is saved in {result_path}")
    print("[info]: Animate over !!!")