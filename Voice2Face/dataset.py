import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import pdb

def load_voice(voice_item):
    voice_data = np.load(voice_item['filepath'])
    voice_data = voice_data.T.astype('float32')
    voice_label = voice_item['label_id']
    return voice_data, voice_label

def load_face(face_item):
    face_data = Image.open(face_item['filepath']).convert('RGB').resize([64, 64])
    face_data = np.transpose(np.array(face_data), (2, 0, 1))
    face_data = ((face_data - 127.5) / 127.5).astype('float32')
    face_label = face_item['label_id']
    return face_data, face_label

class VoiceDataset(Dataset):
    def __init__(self, voice_list, nframe_range):
        self.voice_list = voice_list
        self.crop_nframe = nframe_range[1]

    def __getitem__(self, index):
        voice_data, voice_label = load_voice(self.voice_list[index])
        assert self.crop_nframe <= voice_data.shape[1]
        pt = np.random.randint(voice_data.shape[1] - self.crop_nframe + 1)
        voice_data = voice_data[:, pt:pt+self.crop_nframe]
        return voice_data, voice_label

    def __len__(self):
        return len(self.voice_list)

class FaceDataset(Dataset):
    def __init__(self, face_list, block_len):
        self.face_list = face_list
        self.block_len = block_len

    def __getitem__(self, index):
        face_data, face_label = load_face(self.face_list[index])
        #print (self.face_list[index], face_data.shape)
        mask = self.get_masks()
        mask = np.repeat(np.expand_dims(mask, 0), 3, axis = 0)
        assert mask.shape == face_data.shape
        if np.random.random() > 0.5:
           face_data = np.flip(face_data, axis=2).copy()
        return face_data, face_label, mask
    
    def get_masks(self):
        block_len = self.block_len
        d0_len = d1_len = 64
        d0_min_len = 12
        d0_max_len = d0_len - d0_min_len
        d1_min_len = 12
        d1_max_len = d1_len - d1_min_len
        if block_len == 0:
            d0_mask_len = np.random.randint(d0_min_len, d0_max_len)
            d1_mask_len = np.random.randint(d1_min_len, d1_max_len)
        else:
            d0_mask_len = d1_mask_len = block_len

        d0_start = np.random.randint(0, d0_len - d0_mask_len + 1)
        d1_start = np.random.randint(0, d1_len - d1_mask_len + 1)

        mask = np.zeros((d0_len, d1_len), dtype=np.uint8).astype('float32')
        mask[d0_start:(d0_start + d0_mask_len),
             d1_start:(d1_start + d1_mask_len)] = 1

        mask_loc = d0_start, d1_start, d0_mask_len, d1_mask_len
        
        return mask
        

    def __len__(self):
        return len(self.face_list)
