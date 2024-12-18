import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
from PIL import  ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
from asn.data.datasets.download_datasets import maybe_download_shapeworld4
       
from functools import reduce
from typing import Callable, Iterable, List, Optional, Tuple
from ground_slash.program import Constraint, Naf, Number, PredLiteral, SymbolicConstant
from asn.data.expression import ComplexQuery

def get_encoding(color, shape, shade, size):
    
    if color == 'red':
        col_enc = [1,0,0,0,0,0,0,0,0]
    elif color == 'blue':
        col_enc = [0,1,0,0,0,0,0,0,0]
    elif color == 'green':
        col_enc = [0,0,1,0,0,0,0,0,0]
    elif color == 'gray':
        col_enc = [0,0,0,1,0,0,0,0,0]
    elif color == 'brown':
        col_enc = [0,0,0,0,1,0,0,0,0]
    elif color == 'magenta':
        col_enc = [0,0,0,0,0,1,0,0,0]
    elif color == 'cyan':
        col_enc = [0,0,0,0,0,0,1,0,0]
    elif color == 'yellow':
        col_enc = [0,0,0,0,0,0,0,1,0]
    elif color == 'black':
        col_enc = [0,0,0,0,0,0,0,0,1]
        

    if shape == 'circle':
        shape_enc = [1,0,0,0]
    elif shape == 'triangle':
        shape_enc = [0,1,0,0]
    elif shape == 'square':
        shape_enc = [0,0,1,0]    
    elif shape == 'bg':
        shape_enc = [0,0,0,1]
        
    if shade == 'bright':
        shade_enc = [1,0,0]
    elif shade =='dark':
        shade_enc = [0,1,0]
    elif shade == 'bg':
        shade_enc = [0,0,1]
        
        
    if size == 'small':
        size_enc = [1,0,0]
    elif size == 'big':
        size_enc = [0,1,0]
    elif size == 'bg':
        size_enc = [0,0,1]
    
    return col_enc + shape_enc + shade_enc + size_enc + [1]
    
    
class ShapeWorld4(Dataset):
    def __init__(self, root, mode, ret_obj_encoding=False):
        
        maybe_download_shapeworld4()

        self.ret_obj_encoding = ret_obj_encoding
        self.root = root
        self.mode = mode
        assert os.path.exists(root), 'Path {} does not exist'.format(root)

        #dictionary of the form {'image_idx':'img_path'}
        self.img_paths = {}
        
        
        for file in os.scandir(os.path.join(root, 'images', mode)):
            img_path = file.path
            
            img_path_idx =   img_path.split("/")
            img_path_idx = img_path_idx[-1]
            img_path_idx = img_path_idx[:-4][6:]
            try:
                img_path_idx =  int(img_path_idx)
                self.img_paths[img_path_idx] = img_path
            except:
                print("path:",img_path_idx)
                
        
        count = 0
        
        #target maps of the form {'target:idx': query string} or {'target:idx': obj encoding}
        self.query_map = {}
        self.obj_map = {}
                
        with open(os.path.join(root, 'labels', mode,"world_model.json")) as f:
            worlds = json.load(f)
            
            #iterate over all objects
            for world in worlds:
                num_objects = 0
                target_query = []
                obj_enc = []
                for entity in world['entities']:
                    
                    color = entity['color']['name']
                    shape = entity['shape']['name']
                    
                    shade_val = entity['color']['shade']
                    if shade_val == 0.0:
                        shade = 'bright'
                    else:
                        shade = 'dark'
                    
                    size_val = entity['shape']['size']['x']
                    if size_val == 0.075:
                        size = 'small'
                    elif size_val == 0.15:
                        size = 'big'
                    
                    name = 'o' + str(num_objects+1)
                    
                    q = Constraint(
                        Naf(
                            PredLiteral(
                                "object",
                                *tuple([SymbolicConstant(name),
                                        SymbolicConstant(color),
                                        SymbolicConstant(shape),
                                        SymbolicConstant(shade),
                                        SymbolicConstant(size)]),
                                )
                            )
                        )
                    
                    #target_query = target_query+ ":- not object({},{},{},{},{}). ".format(name, color, shape, shade, size)
                    target_query.append(q)
                    obj_enc.append(get_encoding(color, shape, shade, size))
                    num_objects += 1
                    
                #bg encodings
                for i in range(num_objects, 4):
                    name = 'o' + str(num_objects+1)
                    
                    q = Constraint(
                        Naf(
                            PredLiteral(
                                "object",
                                *tuple([SymbolicConstant(name),
                                        SymbolicConstant(f'black'),
                                        SymbolicConstant(f'bg'),
                                        SymbolicConstant(f'bg'),
                                        SymbolicConstant(f'bg')]),
                                )
                            )
                        )
                    target_query.append(q)                    
                    # target_query = target_query+ ":- not object({},black,bg, bg, bg). ".format(name)
                    obj_enc.append(get_encoding("black","bg","bg","bg"))
                    num_objects += 1

                target_query = ComplexQuery(*target_query)

                self.query_map[count] = target_query
                self.obj_map[count] = np.array(obj_enc)
                count+=1
            
            
                    
    def __getitem__(self, index):
        
        #get the image
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        img = transform(img)
        img = (img - 0.5) * 2.0  # Rescale to [-1, 1].

        if self.ret_obj_encoding:
            return {'im':img}, self.query_map[index] ,self.obj_map[index]
        else:
            return {'im':img}, self.query_map[index]

    def __len__(self):
        return len(self.img_paths)
    
    
    # def to_queries(self, y: Iterable[int]) -> List[Constraint]:
        
        
        
    #     return [
    #         Constraint(
    #             Naf(
    #                 PredLiteral(
    #                     "object",
    #                     *tuple(SymbolicConstant(f"i{i+1}") for i in range(self.n)),
    #                     Number(y_i.item()),
    #                 )
    #             )
    #         )
    #         for y_i in y
    #     ]
