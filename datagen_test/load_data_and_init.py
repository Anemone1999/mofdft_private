import numpy as np
import json
import pyscf
from pyscf import gto, scf, dft, df

with np.load("../data/Ethanol/input/0_test_data.npz",allow_pickle=True) as f:
    data=f["data"][()]
    print(data.keys())
    #print("data['mol'] 的原始类型：", type(data["mol"]))  # 先看是字符串还是其他类型
    mol=json.loads(data["mol"])  # 如果是字符串，尝试用 json.loads() 解析
    print(mol.keys())
    atoms=mol["_atom"]
    basis=mol["basis"]
    charge=mol["charge"]
    spin=mol["spin"]
    scf_summary=data["scf_summary"]

gt_tsxc = 0
gt_etot = sum(data['scf_summary'].values()) - data['scf_summary']['nuc']
gt_coeff = data['rho_coeff_default']
gt_terms = {
    'j': data['scf_summary']['coul'], 'vext': data['scf_summary']['e1'] - data['Ts'],
    'tsxc': gt_tsxc, 'corr': data['Ts'] + data['scf_summary']['exc'] - gt_tsxc
    }

mol = pyscf.gto.Mole.loads(data['mol'])
mol = pyscf.M(atom=mol.atom, basis=mol.basis)

grid = pyscf.dft.gen_grid.Grids(mol)
grid.level = 2
grid.build()

reference_mol = pyscf.M(atom='H 0 0 0; C 0 -0.5 0.5; N 0 0.5 1; O 0 0.5 -0.5; F 0.5 0 0', basis=mol.basis, spin=1)
auxbasis = pyscf.df.aug_etb(reference_mol, 2.5)

auxmol = pyscf.df.addons.make_auxmol(mol, auxbasis=auxbasis)
auxao_values = pyscf.dft.numint.eval_ao(auxmol, grid.coords, deriv=1)

with np.load("rho_data_ethanol.pbe.npz", allow_pickle=True) as f:
    gt_rho = f["gt_rho"]
    grid_weights = f["grid_weights"]
    auxao_values_ref = f["auxao_values"]
    gt_coeff_cpu = f["gt_coeff_cpu"]

np.allclose(auxao_values[0], auxao_values_ref, atol=1e-9)
