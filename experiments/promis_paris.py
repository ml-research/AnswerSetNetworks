# PyTorch
import torch
import torch.optim as optim



#ASN
from asn.asn import ASN
from typing import  Iterable, List
from ground_slash.program import Constraint, Naf, PredLiteral, SymbolicConstant
from asn.models.promis_mock_model import PromisMockNet
from asn.solver import SolvingContext
from ground_slash.grounding import Grounder
from ground_slash.program import Program

#Python libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from time import time 
from tqdm import tqdm

#rtpt
from rtpt import RTPT


parser = argparse.ArgumentParser()

parser.add_argument("--steps", type=int, default=500, help="the number of pixels to solve in each step")
parser.add_argument("--size", type=int, default=6500, help="the image width/height")
parser.add_argument("--title", "--t", type=str, default="promis")
parser.add_argument("--log-path", "--lpath", type=str, default="./logs/")
parser.add_argument("--device", "--d", type=str, default="cuda")
parser.add_argument("--credentials", type=str, default="DO")
parser.add_argument("--data-path", type=str, default="/workspaces/ASN/data/promis_paris/")

args = parser.parse_args()

rtpt = RTPT(name_initials=args.credentials, experiment_name='ProMis PARIS in ASN', max_iterations=1)
rtpt.start()


# # Program
prog_str = '''
% Batch of probabilistic facts
point(x0,y0). 

#npp(over(X,Y,park), [1,0]):- point(X,Y).
#npp(over(X,Y,primary), [1,0]):- point(X,Y).
#npp(over(X,Y,eiffel_tower_stadium), [1,0]):- point(X,Y).
#npp(over(X,Y,bercy_arena), [1,0]):- point(X,Y).
#npp(over(X,Y,secondary), [1,0]):- point(X,Y).
#npp(embassy(X,Y), [1,0]):- point(X,Y).
#npp(government(X,Y), [1,0]):- point(X,Y).

% Its okay to go over park areas
permission(X,Y) :- over(X,Y, park, 1).

% Its okay to fly over major roads
permission(X,Y) :- over(X, Y, primary,1).
permission(X,Y) :- over(X, Y, secondary,1).

% define sport sites
sport_sites(X,Y) :- over(X,Y, eiffel_tower_stadium,0).
sport_sites(X,Y) :- over(X,Y, bercy_arena,0).

% define public buildings
public_building(X,Y) :- embassy(X,Y,0).
public_building(X,Y) :- government(X,Y,0).

%it is not allowed to fly over sport sites and public buildings
permitted(X,Y) :- sport_sites(X,Y).
permitted(X,Y) :- public_building(X,Y).

%we are only permitted to fly over the permitted areas which are not restricted otherwise
airspace(X,Y) :- permission(X,Y), not permitted(X,Y).

'''
# % Query
# :- not landscape(x0), light_drone.
# :- not landscape(x0)
# :- not light_drone.


# ground program
grounder = Grounder(Program.from_string(prog_str))
grnd_prog = grounder.ground()
print("Grounded program:\n",grnd_prog,"\n")

#print("ASN program:\n",prog_str,"\n")
asn = ASN.from_string(prog_str)

#mock network which is not trainable but acts as probabilsitc fact
model = PromisMockNet()
model.to(args.device)

park_data = np.nan_to_num(np.load(args.data_path+"/park.npy"),0)
primary_data = np.load(args.data_path+"/primary.npy")
secondary_data = np.load(args.data_path+"/secondary.npy")
eifeltower_data = np.load(args.data_path+"/eifeltower.npy")
becyarena_data = np.load(args.data_path+"/bercyarena.npy")
government_data = np.load(args.data_path+"/government.npy")
embassy_data = np.load(args.data_path+"/embassy.npy")

    
print("Data loaded")
print("park_data shape:", park_data.shape,
      "primary_data shape:", primary_data.shape,
      "secondary_data shape:", secondary_data.shape,
      "eifeltower_data shape:", eifeltower_data.shape,
      "embassy_data shape:", embassy_data.shape,
      "becyarena_data shape:", becyarena_data.shape)

def process_data(data, path):
    """
    Load and return the data from the numpy file and store it in the visualization folder
    """

    data = torch.tensor(data.flatten()).T
    data = data.clamp_(0, 1)

    print('data min:', data.min())
    print('data max:', data.max())

    if data.isnan().sum() > 0:
        print("path", path)
    return torch.stack((data, 1-data), dim=1).to(device='cuda')


park_data = process_data(park_data, "park")
primary_data = process_data(primary_data, "primary")
secondary_data = process_data(secondary_data, "secondary")
eifeltower_data = process_data(eifeltower_data, "eifeltower")
government_data = process_data(government_data, "government")
embassy_data = process_data(embassy_data, "embassy")
becyarena_data = process_data(becyarena_data, "bercyarena")
print("Data processed into tensors")
npp_rule_dict={}


total_steps = args.size**2 # 6500  = 262144

epochs = int(total_steps / args.steps) # 262144 / 512 = 512 or 262144 / 256 = 1024
assert total_steps % args.steps == 0, "total steps must be divisible by steps"


print("total steps:", total_steps)
print("epochs:", epochs)
print("steps:", args.steps)
print("device:",args.device)


asn.configure_NPPs(
        {
            npp_rule: {
                "model": model,
                "optimizer": optim.Adam(model.parameters(), lr=0.1)
                if not i
                else None,
            }
            for i, npp_rule in enumerate(asn.rg.npp_edges)
        }
)   
        

def get_npp_data_dict(i, size):
    """
    Returns a chunk of pixels from map to be processed.
    """
    npp_data_dict_str={}
    npp_data_dict_str['#npp(over(x0,y0,park),[1,0]) :- point(x0,y0).']= [park_data[i*size:(i+1)*size,:]]
    npp_data_dict_str['#npp(over(x0,y0,primary),[1,0]) :- point(x0,y0).']= [primary_data[i*size:(i+1)*size,:]]
    npp_data_dict_str['#npp(over(x0,y0,secondary),[1,0]) :- point(x0,y0).']= [secondary_data[i*size:(i+1)*size,:]]
    npp_data_dict_str['#npp(over(x0,y0,eiffel_tower_stadium),[1,0]) :- point(x0,y0).']= [eifeltower_data[i*size:(i+1)*size,:]]
    npp_data_dict_str['#npp(government(x0,y0),[1,0]) :- point(x0,y0).']= [government_data[i*size:(i+1)*size,:]]
    npp_data_dict_str['#npp(embassy(x0,y0),[1,0]) :- point(x0,y0).']= [embassy_data[i*size:(i+1)*size,:]]  
    npp_data_dict_str['#npp(over(x0,y0,bercy_arena),[1,0]) :- point(x0,y0).']= [becyarena_data[i*size:(i+1)*size,:]] 
    
    npp_data_dict={}
    for e in asn.rg.npp_edges:
        npp_data_dict[e] = npp_data_dict_str[str(e)]
    return npp_data_dict
    

def to_queries(y: Iterable[int]) -> List[Constraint]:
    return [
        Constraint(
            Naf(
                PredLiteral(
                    "airspace",
                    *tuple((SymbolicConstant(f"x0"), SymbolicConstant(f"y0")))
                ),
            )
        )
        for y_i in y
    ]
    

start = time()

#store the solved pixels for later stacking
solved_pixels = torch.zeros([total_steps])
solving_steps = 0 

#epochs are not epochs as used in the DL sense but rather the chunks of pixels to solve
for i in tqdm(range(epochs)):

    npp_data_dict = get_npp_data_dict(i, args.steps)
    queries = to_queries(y=torch.ones(args.steps))
    rg = asn.encode_queries(queries)
    
    # NPP forward pass
    npp_ctx_dict = asn.npp_forward(
        npp_data={
            npp_rule: (npp_data_dict[npp_rule][0],)
            for i,npp_rule in enumerate(asn.rg.npp_edges)
        },
    )

    # initialize solving context
    solving_ctx = SolvingContext(
        len(queries),
        npp_ctx_dict,
    )

    for phase in range(1):

        # prepare graph block
        graph_block = asn.prepare_block(
            queries=queries,
            rg=rg,  # pass pre-computed reasoning graph
            phase=phase,
            device=args.device
        )

        # solve graph block
        graph_block = asn.solve(graph_block)


        # update stable models
        solving_ctx.update_SMs(graph_block)
          
        Q = solving_ctx.p_Q
        solved_pixels[solving_steps*args.steps:(solving_steps+1)*args.steps]= Q.squeeze()
        solving_steps += 1


end = time()
total_time = end - start

log_path = Path(args.log_path)
log_path.mkdir(parents=True, exist_ok=True)

#create log file
exp_log = {"title": args.title,
        "time": total_time,
        "steps": args.steps,
        "epochs": epochs,
        "image_size": args.size,
        "total_steps": total_steps}

with Path(log_path,f"{args.title}.json").open("w") as f:
    json.dump(exp_log, f, indent=4)


#stack all pixels
#pixels_stacked = torch.stack(solved_pixels)
pixel_image = solved_pixels.view(args.size,args.size).cpu().numpy()
plt.imshow(pixel_image)

path_folder = Path(log_path,"npy")
path_folder.mkdir(parents=True, exist_ok=True)
path_img = Path(log_path,"img")
path_img.mkdir(parents=True, exist_ok=True)

#save generated map
path_img = Path(log_path,"img","landscape_{}.png".format(args.title))
path_npy =Path(log_path,"npy","landscape_{}.npy".format(args.title))
plt.savefig(path_img)

np.save(path_npy, pixel_image)
print("saved image to", path_img)

