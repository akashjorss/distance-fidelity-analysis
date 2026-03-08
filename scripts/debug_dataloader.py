import sys, os
sys.path.insert(0, '/gpfs/workdir/malhotraa/gsplat/examples')
from datasets.colmap import Parser, Dataset
import torch

p = Parser('/gpfs/workdir/malhotraa/data/LLFF/fern', factor=4, normalize=True, test_every=8)
print('Parser loaded:', len(p.image_names), 'images')
print('image_paths:', len(p.image_paths))

ti = [0,1,2,3,4,5,6,7,8,9]
trainset = Dataset(p, split='train', train_indices=ti)
valset = Dataset(p, split='val', train_indices=ti)
print('trainset:', len(trainset), 'valset:', len(valset))
print('train indices:', trainset.indices)
print('val indices:', valset.indices)

# Test DataLoader with num_workers=0
dl = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
for i, data in enumerate(dl):
    print('batch', i, 'image shape', data['image'].shape)
    if i >= 3:
        break
print('DataLoader OK with num_workers=0')

# Test with num_workers=4
try:
    dl2 = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    for i, data in enumerate(dl2):
        print('batch_mp', i, 'image shape', data['image'].shape)
        if i >= 3:
            break
    print('DataLoader OK with num_workers=4')
except Exception as e:
    print('DataLoader num_workers=4 FAILED:', e)
