import cv2
import torch
from os.path import join
from tqdm import tqdm
    

class InferenceDataset():
    def __init__(self, frame_list, crop_image_path, mouth_size):
        
        self.frame_list = frame_list
        self.crop_image_path = crop_image_path
        self.mouth_size = mouth_size


    def make_dataset(self, full_frame_list, crop_image_path):
        mouth_mask = []
        mouth_contour = []
        referece = []
        for frame in full_frame_list:
            basename = frame.replace('.png', '')
            
            mask_frame_path = join(crop_image_path, basename + '_mask.png')
            mean_mask_frame_path = join(crop_image_path, basename + '_mean_mask.png')
            # fixed or not reference image
            # ref_frame_path = random.choice(glob(join(align_image_path, basename + '_image.png')))
            ref_frame_path = join(crop_image_path, basename + '_image.png')
            
            mask_frame = cv2.imread(mask_frame_path)
            mean_mask_frame = cv2.imread(mean_mask_frame_path)
            ref_frame = cv2.imread(ref_frame_path)
            
            mouth_mask.append(mask_frame)
            mouth_contour.append(mean_mask_frame)
            referece.append(ref_frame)
        return mouth_mask, mouth_contour, referece
        
    def __getitem__(self, index):
        mouth_mask, mouth_contour, referece = self.make_dataset(self.frame_list, self.crop_image_path)
        
        mouth_mask_img = cv2.resize(mouth_mask[index], (self.mouth_size, self.mouth_size)) / 255.
        mouth_contour_img = cv2.resize(mouth_contour[index], (self.mouth_size, self.mouth_size)) / 255.
        referece_img = cv2.resize(referece[index], (self.mouth_size, self.mouth_size)) / 255.

        mouth_mask_img = torch.from_numpy(mouth_mask_img).permute(2, 0, 1).float().cuda()
        mouth_contour_img = torch.from_numpy(mouth_contour_img).permute(2, 0, 1).float().cuda()
        referece_img = torch.from_numpy(referece_img).permute(2, 0, 1).float().cuda()
        
        return mouth_mask_img, mouth_contour_img, referece_img
        
    def __len__(self):
        return len(self.frame_list)