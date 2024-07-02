from ctypes import set_errno
import sys
import os
from renormalizer.model import Op
from renormalizer.model import basis as ba
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.sbm import OhmicSDF
from renormalizer.utils import EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.utils import log
from renormalizer.tn import BasisTree, TTNO, TTNS, TreeNodeBasis
import argparse
import numpy as np
from renormalizer.utils.log import package_logger
# import pickle

if __name__ == '__main__':
    # argpaser
    # 
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", help="sbm s", type=float)
    parser.add_argument("--alpha", help="sbm \alpha coupling", type=float)
    parser.add_argument("--omega_c", help="sbm \omega_c * \Omega", type=float)
    parser.add_argument("--nsteps", help="tdvp step", type=int)
    parser.add_argument("--omega", help="sbm \Omega tunnelling spliting", type=float)
    parser.add_argument("--nmodes", help="sbm env modes", type=int)
    parser.add_argument("--bond_dims", help="mps/tns bond dim", type=int)
    parser.add_argument("--td_method", help="0: tdvp_ps, 1: tdvp_ps2, 3: tdvp_vmf, 4: prop_and_compress_tdrk4", type=int)
    args = parser.parse_args()
    s = args.s
    alpha = args.alpha
    omega_c  = args.omega_c
    nsteps = args.nsteps # was 200

    nmodes = args.nmodes
    bond_dims = args.bond_dims
    
    Omega = args.omega
    omega_c_reno  = omega_c * Omega

    td_method = args.td_method
    # parm translate
    s_reno = s
    alpha_reno = 4*alpha # tranlate from wang1 to PRL

    # set log
    logger = package_logger

    job_name = f"traj_s{s:.2f}_alpha{alpha:.2f}_Omega{Omega:.2f}_omega_c{omega_c:.2f}_nmodes{nmodes}_bond_dims{bond_dims}"  ####################

    dump_dir = job_name
    os.makedirs(job_name, exist_ok=True)
    log.register_file_output(os.path.join(dump_dir, f'{job_name}.log'), mode="w")

    # use old settings from example
    # nmodes = 1000
    Ms = bond_dims
    upper_limit = 30

    Delta = Omega * 0.5
    eps = 0
    # call a sdf
    sdf = OhmicSDF(alpha_reno, omega_c_reno, s_reno)


    # wang1?
    w, c2 = sdf.Wang1(nmodes)
    c = np.sqrt(c2)
    logger.info(f"w:{w}")
    logger.info(f"c:{c}")

    # reno ?
    reno = sdf.reno(w[-1])
    logger.info(f"renormalization constant: {reno}")
    Delta *= reno

    ham_terms = []



    # h_s
    # sigma_z sigma_x
    ham_terms.extend([Op("sigma_z","spin",factor=eps, qn=0),
            Op("sigma_x","spin",factor=Delta, qn=0)])
    # ham_terms.append(Op("sigma_x","spin",factor=Delta, qn=0))

    # boson energy
    for imode in range(nmodes):
        op1 = Op(r"p^2",f"v_{imode}",factor=0.5, qn=0)
        op2 = Op(r"x^2",f"v_{imode}",factor=0.5*w[imode]**2, qn=0)
        ham_terms.extend([op1,op2])

    # system-boson coupling
    for imode in range(nmodes):
        op = Op(r"sigma_z x", ["spin", f"v_{imode}"],
                factor=0.5*c[imode], qn=[0,0])
        ham_terms.append(op)

    # set basis
    nbas = np.max([16 * c2/w**3, np.ones(nmodes)*4], axis=0)
    nbas = np.round(nbas).astype(int)
    logger.info(nbas)
    basis = [ba.BasisHalfSpin("spin",[0,0])]
    for imode in range(nmodes):
        basis.append(ba.BasisSHO(f"v_{imode}", w[imode], int(nbas[imode])))


    tree_order = 2
    basis_vib = basis[1:]
    elementary_nodes = []


    root = BasisTree.binary_mctdh(basis_vib, contract_primitive=True, contract_label=nbas>Ms, dummy_label="n").root

    root.add_child(TreeNodeBasis(basis[:1]))

    basis_tree = BasisTree(root)
    basis_tree.print(print_function=logger.info)

    # basis_tree = BasisTree.linear(basis)
    ttno = TTNO(basis_tree, ham_terms)
    exp_z = TTNO(basis_tree, Op("sigma_z", "spin"))
    # exp_x = TTNO(basis_tree, Op("sigma_x", "spin"))
    ttns = TTNS(basis_tree)
    ttns.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=Ms)
    ttns = expand_bond_dimension_general(ttns, ttno, ex_mps=None)
    logger.info(ttns.bond_dims)
    logger.info(ttno.bond_dims)
    logger.info(len(ttns))

    # py310 only 
    # match td_method:
    #     case 0: 
    #         ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    #     case 1:
    #         ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps2)
    #     case 2:
    #         ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_vmf)
    #     case 3:
    #         ttns.evolve_config = EvolveConfig(EvolveMethod.prop_and_compress_tdrk4)
    #     case _:
    #         ttns.evolve_config = EvolveConfig(EvolveMethod.prop_and_compress_tdrk4)
    if td_method == 0:
        ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    elif td_method == 1:
        ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps2)
    elif td_method == 2:
        ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_vmf)
    elif td_method == 3:
        ttns.evolve_config = EvolveConfig(EvolveMethod.prop_and_compress_tdrk4)
    else :
        ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)

    # old settings
    # nsteps = args.nsteps # was 200
    # 
    # simulation_time = 10 * Omega
    dt = 0.1/Omega # 
    
    expectations = []

    for i in range(nsteps):
        logger.info(f'proceeding step {i}')
        ttns = ttns.evolve(ttno, dt)
        if i == nsteps-1:
            dump_file = os.path.join(dump_dir, f'{job_name}_last_step_ttns.npz')
            ttns.dump(dump_file)
        z = ttns.expectation(exp_z)
        expectations.append(z)
        
        logger.info(z)

    logger.info(expectations)
    with open(os.path.join(dump_dir, f'{job_name}_p.xvg'), 'w') as f:
        for inf in range(len(expectations)):
            f.write(f'{inf}  {expectations[inf]} \n')
    with open(os.path.join(dump_dir, f'{job_name}_1-p_baselog10.xvg'), 'w') as f:
        for inf in range(len(expectations)):
            f.write(f'{inf}  {np.log10(1-expectations[inf])} \n')
            