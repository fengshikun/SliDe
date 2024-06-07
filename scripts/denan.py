import torch
import sys
sys.path.append(sys.path[0]+'/..')
from torchmdnet.models.torchmd_etf2d import EMB

if __name__ == '__main__':
    num_spherical=3
    num_radial=6
    envelope_exponent=5
    int_emb_size=64
    cutoff_upper = 5
    emb = EMB(num_spherical, num_radial, cutoff_upper, envelope_exponent).cuda()
    dist = torch.load('dist.pt')
    angle = torch.load('angle.pt')
    torsion = torch.load('torsion.pt')
    idx_kj = torch.load('idx_kj.pt')
    embdding = emb(dist, angle, torsion, idx_kj)
    print(embdding.isnan().sum())