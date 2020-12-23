from __future__  import print_function
import os
import json
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import random

import h5py

class ShapeNetLoader(data.Dataset):
    def __init__(self, data_dump_folder, indim=2, freeze_data=False, id=None, require_normal=False, num_pts=1024, mode="train", jitter_type="None"):
        super().__init__()
        assert mode in ["train", "valid", "test"]
        mode = "valid" if mode == "test" else mode # only valid set is available as the test 

        if freeze_data:
            # freeze test data
            print("Freezing data. No randomness!")
            transform_fn = os.path.join(
                data_dump_folder, "random_T_100k_uni-180-0.2.h5")
            with h5py.File(transform_fn, "r") as f:
                self.test_transforms = np.asarray(f["transform"])
                # self.test_pose1s = np.asarray(f["pose1"])
            self.fix_idx = True
        else:
            self.test_transforms = None
            self.fix_idx = False 

        self.mode = mode
        self.jitter_type = jitter_type
        self.num_pts = num_pts
        self.require_normal = require_normal
        data_dump_folder = os.path.join(
            data_dump_folder, "ShapeNetAtlasNetH5"
        )

        # get h5path
        fn_cat = os.path.join(data_dump_folder, f"{mode}_cat.txt")
        h5path = [line.rstrip() for line in list(open(fn_cat, "r"))]
        if id is not None and id != -1:
            h5path = h5path[id:id+1]
        if indim == 2:
            h5path = ["plane_2d.h5"]
        self.h5path = [
            os.path.join(data_dump_folder, mode, h5path_) for h5path_ in h5path]
        self.dataset = [h5py.File(path, "r") for path in self.h5path]

        # get len and cumsum
        lens = [len(d["pcd"]["point"]) for d in self.dataset]
        self.len = np.sum(lens)
        self.cuts = np.cumsum(lens)

        # close file
        for i, d in enumerate(self.dataset):
            d.close()
            self.dataset[i] = None

    def preprocess(self, xyz, other=None):
        # xyz: Nx3, other(NxC)
        # Sampling
        if self.fix_idx:
            idx = np.arange(self.num_pts)
        else:
            idx = np.random.choice(
                len(xyz), self.num_pts, replace=False)
        xyz = xyz[idx]
        if other is None:
            return xyz 
        else:
            other = other[idx]
            return xyz, other


    def __len__(self):
        return self.len 


    def __getitem__(self, item):
        d_index = np.searchsorted(self.cuts, item + 1) # Figure out the dataset index
        original_item = item
        item = (item - self.cuts[d_index - 1]) if d_index > 0 else item
        if self.dataset[d_index] is None:
            self.dataset[d_index] = h5py.File(self.h5path[d_index], "r")
        xyz = np.asarray(
            self.dataset[d_index]["pcd"]["point"][str(item)]).astype(np.float32)
        # label = self.dataset[d_index]["pcd"]["point"]

        if self.require_normal:
            normal = np.asarray(
                self.dataset[d_index]["pcd"]["normal"][str(item)]).astype(np.float32)

        # Sampling points
        if self.require_normal:
            xyz_resample, normal_resample = self.preprocess(xyz, normal)
        else:
            xyz_resample = self.preprocess(xyz)

        data = {
            "pc": xyz_resample.astype("float32"),
            # "pc": xz,
            "label": d_index}
        if self.require_normal:
            data["normal"] = normal_resample.astype("float32")
        if self.test_transforms is not None:
            transform = self.test_transforms[original_item] # 4x4 trans mat
            data["transform"] = transform
            data["item_idx"] = original_item 
            
        return data 


# Dataset preparation code from AtlasNetV2 code
CHUNK_SIZE = 150
lenght_line = 60
def my_get_n_random_lines(path, n=5):
    MY_CHUNK_SIZE = lenght_line * (n+2)
    lenght = os.stat(path).st_size
    with open(path, 'r') as file:
            file.seek(random.randint(400, lenght - MY_CHUNK_SIZE))
            chunk = file.read(MY_CHUNK_SIZE)
            lines = chunk.split(os.linesep)
            return lines[1:n+1]

class ShapeNetLoaderDeprecated(data.Dataset):
    # def __init__(self, train=True, options=None):
    def __init__(self, data_dump_folder, num_pts=1024, mode="train", jitter_type="None"):
        train = mode == "train"
        data_dump_folder = os.path.join(
            data_dump_folder, "ShapeNet"
        )
        rootimg = os.path.join(
            data_dump_folder, "ShapeNet/ShapeNetRendering")
        rootpc = os.path.join(
            data_dump_folder, "customShapeNet")

        class_choice = None
        npoints = num_pts 
        normal = False
        balanced = False
        gen_view=False
        SVR=False
        idx=0
        self.balanced = balanced
        self.normal = normal
        self.train = train
        self.rootimg = rootimg
        self.rootpc = rootpc
        self.npoints = npoints
        self.datapath = []
        self.catfile = os.path.join(
            data_dump_folder, 'synsetoffset2category.txt')
        self.cat = {}
        self.meta = {}
        self.SVR = SVR
        self.gen_view = gen_view
        self.idx=idx
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        print(self.cat)
        empty = []
        for item in self.cat:
            dir_img  = os.path.join(self.rootimg, self.cat[item])
            fns_img = sorted(os.listdir(dir_img))

            try:
                dir_point = os.path.join(self.rootpc, self.cat[item], 'ply')
                fns_pc = sorted(os.listdir(dir_point))
            except:
                fns_pc = []
            fns = [val for val in fns_img if val + '.points.ply' in fns_pc]
            print('category ', self.cat[item], 'files ' + str(len(fns)), len(fns)/float(len(fns_img)), "%"),
            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]


            if len(fns) != 0:
                self.meta[item] = []
                for fn in fns:
                    objpath = "./data/ShapeNetCorev2/" +  self.cat[item] + "/" + fn + "/models/model_normalized.ply"
                    self.meta[item].append( ( os.path.join(dir_img, fn, "rendering"), os.path.join(dir_point, fn + '.points.ply'), item, objpath, fn ) )
            else:
                empty.append(item)
        for item in empty:
            del self.cat[item]
        self.idx2cat = {}
        self.size = {}
        i = 0
        for item in self.cat:
            self.idx2cat[i] = item
            self.size[i] = len(self.meta[item])
            i = i + 1
            # for fn in self.meta[item]:
            l = int(len(self.meta[item]))
            for fn in self.meta[item][0:l]:
                self.datapath.append(fn)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.transforms = transforms.Compose([
                             transforms.Resize(size =  224, interpolation = 2),
                             transforms.ToTensor(),
                             # normalize,
                        ])

        # RandomResizedCrop or RandomCrop
        self.dataAugmentation = transforms.Compose([
                                         transforms.RandomCrop(127),
                                         transforms.RandomHorizontalFlip(),
                            ])
        self.validating = transforms.Compose([
                        transforms.CenterCrop(127),
                        ])

        # self.perCatValueMeter = {}
        # for item in self.cat:
        #     self.perCatValueMeter[item] = AverageValueMeter()
        # self.perCatValueMeter_metro = {}
        # for item in self.cat:
        #     self.perCatValueMeter_metro[item] = AverageValueMeter()
        self.transformsb = transforms.Compose([
                             transforms.Resize(size =  224, interpolation = 2),
                        ])

    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(fn[1]) as fp:
            for i, line in enumerate(fp):
                if i == 2:
                    try:
                        lenght = int(line.split()[2])
                    except ValueError:
                        print(fn)
                        print(line)
                    break
        for i in range(15): #this for loop is because of some weird error that happens sometime during loading I didn't track it down and brute force the solution like this.
            try:
                mystring = my_get_n_random_lines(fn[1], n = self.npoints)
                point_set = np.loadtxt(mystring).astype(np.float32)
                break
            except ValueError as excep:
                print(fn)
                print(excep)

        # centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) #Useless because dataset has been normalised already
        # point_set[:,0:3] = point_set[:,0:3] - centroid
        if not self.normal:
            point_set = point_set[:,0:3]
        else:
            point_set[:,3:6] = 0.1 * point_set[:,3:6]
        point_set = torch.from_numpy(point_set)

        # load image
        if self.SVR:
            if self.train:
                N_tot = len(os.listdir(fn[0])) - 3
                if N_tot==1:
                    print("only one view in ", fn)
                if self.gen_view:
                    N=0
                else:
                    N = np.random.randint(1,N_tot)
                if N < 10:
                    im = Image.open(os.path.join(fn[0], "0" + str(N) + ".png"))
                else:
                    im = Image.open(os.path.join(fn[0],  str(N) + ".png"))

                im = self.dataAugmentation(im) #random crop
            else:
                if self.idx < 10:
                    im = Image.open(os.path.join(fn[0], "0" + str(self.idx) + ".png"))
                else:
                    im = Image.open(os.path.join(fn[0],  str(self.idx) + ".png"))
                im = self.validating(im) #center crop
            data = self.transforms(im) #scale
            data = data[:3,:,:]
        else:
            data = 0
        data = {
            "pc": point_set.contiguous(), 
            }
        return data 


    def __len__(self):
        return len(self.datapath)
    

def extract_h5(mode="train"):
    data_dump_folder = "data_dump"
    train = mode == "train"
    data_dump_folder = os.path.join(
        data_dump_folder, "ShapeNet"
    )
    rootimg = os.path.join(
        data_dump_folder, "ShapeNet/ShapeNetRendering")
    rootpc = os.path.join(
        data_dump_folder, "customShapeNet")

    idx=0
    catfile = os.path.join(
        data_dump_folder, 'synsetoffset2category.txt')
    cat = {}
    meta = {}
    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]
    empty = []
    for item in cat:
        dir_img  = os.path.join(rootimg, cat[item])
        fns_img = sorted(os.listdir(dir_img))

        try:
            dir_point = os.path.join(rootpc, cat[item], 'ply')
            fns_pc = sorted(os.listdir(dir_point))
        except:
            fns_pc = []
        fns = [val for val in fns_img if val + '.points.ply' in fns_pc]
        print('category ', cat[item], 'files ' + str(len(fns)), len(fns)/float(len(fns_img)), "%"),
        if train:
            fns = fns[:int(len(fns) * 0.8)]
        else:
            fns = fns[int(len(fns) * 0.8):]

        if len(fns) != 0:
            meta[item] = []
            for fn in fns:
                objpath = "./data/ShapeNetCorev2/" +  cat[item] + "/" + fn + "/models/model_normalized.ply"
                meta[item].append( ( os.path.join(dir_img, fn, "rendering"), os.path.join(dir_point, fn + '.points.ply'), item, objpath, fn ) )
        else:
            empty.append(item)
    for item in empty:
        del cat[item]
    idx2cat = {}
    size = {}
    i = 0
    data_dump_dir = f"data_dump/ShapeNetAtlasNetH5/{mode}"
    if not os.path.exists(data_dump_dir):
        os.makedirs(data_dump_dir)

    for item in cat:
        datapath = []
        idx2cat[i] = item

        size[i] = len(meta[item])
        i = i + 1
        # for fn in self.meta[item]:
        l = int(len(meta[item]))
        for fn in meta[item][0:l]:
            datapath.append(fn)
        
        vs = []
        ns = []
        for fn in datapath:
            # read ply file
            v, _, n, _ = pcu.read_ply(fn[1]) # return xyz and normal
            if len(v) != 30000:
                print(f"len: {len(v)}, fn: {fn}")
            vs += [v.astype(np.float32)]
            ns += [n.astype(np.float32)]
        # vs = np.stack(vs).astype(np.float32)
        # ns = np.stack(ns).astype(np.float32)
        h5fn = os.path.join(data_dump_dir, f"{item}.h5")
        print(f"writing h5 for: {item}_{mode}, {len(vs)}")
        with h5py.File(h5fn, "w") as f:
            pcd = f.create_group("pcd")
            pcd_point = pcd.create_group("point")
            pcd_normal = pcd.create_group("normal")
            for i in range(len(vs)):
                pcd_point[str(i)] = vs[i]
                pcd_normal[str(i)] = ns[i]
            


if __name__ == "__main__":
    for mode in ["train", "valid"]:
        extract_h5(mode)