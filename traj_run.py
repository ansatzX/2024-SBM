from typing import List, Dict, Tuple, Union, Callable, Any
from ctypes import set_errno
import sys
import os
from renormalizer.model import Op
from renormalizer.model import basis as ba
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.sbm import OhmicSDF
from renormalizer.utils import EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, Quantity
from renormalizer.utils import log
from renormalizer.tn import BasisTree, TTNO, TTNS, TreeNodeBasis
import argparse
import numpy as np
from renormalizer.utils.log import package_logger
import pickle



class rho_ohmic(OhmicSDF):

    def __init__(self, alpha: float, omega_c: Union[Quantity, float], s: float = 1, rho_type: int = 0):
        super().__init__(alpha, omega_c, s)
        self.rho_type = rho_type

    def _dos_Wang1(self, nb, omega_value):
        if self.rho_type == 0:
            return super()._dos_Wang1(nb, omega_value)
        elif self.rho_type == 1:
            rho=np.zeros(nb)
            for i in range(nb):
                if i == 0:
                    rho[i] = 1/(omega_value[i]-0)
                else:
                    rho[i] = 1/(omega_value[i]-omega_value[i-1])

            return rho
    def Wang1(self, nb):
        r"""
        Wang's 1st scheme discretization
        """
        if self.rho_type == 0:
            omega_value = np.array([-np.log(-float(j) / (nb + 1) + 1.0) * self.omega_c for j in range(1, nb + 1, 1)])
        elif self.rho_type == 1:
            # make same range of rho_type 1
            over_length = -np.log(-float(nb) / (nb + 1) + 1.0) * self.omega_c
            omega_value = np.array([over_length/nb *(i+1) for i in range(nb)])
        # general form
        c_j2 = 2.0 / np.pi * omega_value * self.func(omega_value) / self._dos_Wang1(nb, omega_value)

        return omega_value, c_j2
    
if __name__ == '__main__':
    # argpaser
    # 
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", help="sbm s", type=float)
    parser.add_argument("--alpha", help="sbm \alpha coupling", type=float)
    parser.add_argument("--omega_c", default=10,help="sbm \omega_c * \Omega", type=int)
    parser.add_argument("--omega", default=1,help="sbm \Omega tunnelling spliting", type=int)

    parser.add_argument("--nsteps", default=100, help="tdvp step", type=int)
    parser.add_argument("--nmodes",  default=1000,help="sbm env modes", type=int)
    parser.add_argument("--bond_dims", default=20, help="mps/tns bond dim", type=int)
    parser.add_argument("--td_method", default=0, help="0: tdvp_ps, 1: tdvp_ps2, 3: tdvp_vmf, 4: prop_and_compress_tdrk4", type=int)
    parser.add_argument("--store_all", default=0, help="0: n, 1: y", type=int)
    parser.add_argument("--calc_1sites_entropy", default=0, help="0: n, 1: y", type=int)
    parser.add_argument("--calc_mutual_info", default=0, help="0: n, 1: y", type=int)
    parser.add_argument("--rho_type", default=0, help="select rho types in discre..", type=int)
    parser.add_argument("--restart", default=0, help="0: n, 1: y", type=int)





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
    store_all = args.store_all
    is_calc_1sites_entropy = args.calc_1sites_entropy
    is_calc_mutual_info = args.calc_mutual_info
    is_restart = args.restart
    rho_type  = args.rho_type
    # parm translate
    s_reno = s
    alpha_reno = 4*alpha # tranlate from wang1 to PRL

    # set log
    logger = package_logger

    job_name = f"traj_s{s:.2f}_alpha{alpha:.2f}_Omega{Omega}_omega_c{omega_c}_nmodes{nmodes}_bond_dims{bond_dims}_td_method_{td_method}_rho_type_{rho_type}"  ####################

    dump_dir = job_name
    os.makedirs(job_name, exist_ok=True)
    log.register_file_output(os.path.join(dump_dir, f'{job_name}.log'), mode="w")

    # use old settings from example
    # nmodes = 1000
    Ms = bond_dims
    # upper_limit = 30

    Delta = Omega * 0.5
    eps = 0
    # call a sdf
    sdf = rho_ohmic(alpha_reno, omega_c_reno, s_reno, rho_type)


    # wang1?
    w, c2 = sdf.Wang1(nmodes)
    c = np.sqrt(c2)
    logger.info(f"w:{w}")
    logger.info(f"c:{c}")
    with open(os.path.join(dump_dir, f'sdf_wang1_omega.pickle'), 'wb') as f:
        pickle.dump(w, f)
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
    
    basis = [ba.BasisHalfSpin("spin",[0,0])]
    logger.info(f'HalfSpin dof: spin')
    logger.info(f'SHO dof: v_imode')
    for imode in range(nmodes):
        basis.append(ba.BasisSHO(f"v_{imode}", w[imode], int(nbas[imode])))
    logger.info(f'SHO nbas: {nbas.tolist()}')

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
    logger.info(f'ttns.bond_dims: {ttns.bond_dims}')
    logger.info(f'ttno.bond_dims: {ttno.bond_dims}')
    logger.info(f'ttns_length: {len(ttns)}')

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
    logger.info(f'ttns.evolve_config: {ttns.evolve_config}')
    # old settings
    # nsteps = args.nsteps # was 200
    # 
    # simulation_time = 10 * Omega
    dt: float = 0.1/Omega # 
    logger.info(f'dt: {dt}')
    expectations: List[Union[float, complex]] = []
    entropy_1sites_traj: List[Dict[Union[int, List], float]] = []
    mutual_info_traj =[]
    logger.info("ttns.basis.dof_list")
    logger.info(ttns.basis.dof_list)
    logger.info("ttns.basis.dof2idx")
    logger.info(ttns.basis.dof2idx)
    for i in range(nsteps):
        logger.info(f'proceeding step {i}')
        ttns = ttns.evolve(ttno, dt)
        # if store_all or i%10 ==0:
        #     dump_file = os.path.join(dump_dir, f'{job_name}_{i}_step_ttns.npz')
        #     ttns.dump(dump_file)
        # else:
        #     if i == nsteps-1:
        #         dump_file = os.path.join(dump_dir, f'{job_name}_last_step_ttns.npz')
        #         ttns.dump(dump_file)
        # prop calc
        z: Union[float, complex] = ttns.expectation(exp_z)
        # entropy_1sites = []
        # for dof in ttns.basis.dof_list:
        #     entropy_1site: Any = ttns.calc_1dof_entropy(dof)
        #     # ttns.calc_2dof_entropy()
        #     entropy_1sites.append(entropy_1sites)     
        if is_calc_1sites_entropy  : 
            dofs = [f'v_{i}' for i in range(nmodes) ]
            dofs.append('spin')
            logger.info(f'dofs:, {dofs}')
            entropy_1sites: Dict[Union[int, List], float] = ttns.calc_1dof_entropy(dofs)

            with open(os.path.join(dump_dir, f'{i:04}_step_entropy_1site.pickle'), 'wb') as f:
                pickle.dump(entropy_1sites, f)
            entropy_1sites_traj.append(entropy_1sites)
            logger.info(f'step {i} entropy_1sites: {entropy_1sites}')

        if is_calc_mutual_info:
            # mutual_infos = {}
            # entropy_ts = {}
            dof1 ='spin'
            dofs = [(dof1, f'v_{i*10}')for i in range(100)]
            rdm_2dof = ttns.calc_2dof_rdm(dofs)
            mutual_infos, entropy_tuple = ttns.calc_2dof_mutual_info(dofs, rdm_2dof)
            with open(os.path.join(dump_dir, f'{i:04}_step_mutual_infos.pickle'), 'wb') as f:
                pickle.dump(mutual_infos, f) 
            # with open(os.path.join(dump_dir, f'{i:04}_step_rdm_2dofs.pickle'), 'wb') as f:
            #     pickle.dump(rdm_2dof, f) 
            mutual_info_traj.append(mutual_infos)
            logger.info(f'step {i} mutual_infos: {mutual_infos}')  

        expectations.append(z)
        logger.info(f'step {i} z: {z}')
        

    with open(os.path.join(dump_dir, f'expectations.pickle'), 'wb') as f:
        pickle.dump(expectations, f)
    if is_calc_1sites_entropy:
        with open(os.path.join(dump_dir, f'entropy_1sites_traj.pickle'), 'wb') as f:
            pickle.dump(entropy_1sites_traj, f)
    logger.info('expectations')
    logger.info(expectations)

    with open(os.path.join(dump_dir, f'{job_name}_p.xvg'), 'w') as f:
        for inf in range(len(expectations)):
            f.write(f'{inf}  {expectations[inf]} \n')
    
    with open(os.path.join(dump_dir, f'{job_name}_1-p_baselog10.xvg'), 'w') as f:
        for inf in range(len(expectations)):
            f.write(f'{inf}  {np.log10(1-expectations[inf])} \n')

    # with open(os.path.join(dump_dir, f'{job_name}_entropy_1sites.xvg'), 'w') as f:
    #     f.write(f"@ title 's{s:.2f}_alpha{alpha:.2f}' \n")
    #     idxs = sorted(list(ttns.basis.dof2idx.values()))
    #     for i in range(len(ttns.basis.dof2idx)):
    #         f.write(f"@ s{i} legend 's{s:.2f}_alpha{alpha:.2f}_dof_{idxs[i]}' \n")

        # for inf in range(len(entropy_1sites_traj)):
        #     f.write(f'{inf} ')
        #     for dof_idx in idxs:
        #         f.write(f'{entropy_1sites_traj[inf][dof_idx]} ')
        #     f.write(f' \n')

    logger.info(f'done calc, {job_name}')