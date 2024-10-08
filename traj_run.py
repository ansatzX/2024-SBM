from copy import deepcopy
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
import json


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
    
def read_query_config(file):
    with open (file, 'rb') as f:
        dat = pickle.load(f)
    return dat

def check_onestep(timestamps, current_time, future_time, rtol=0, atol=1e-2):

    static_steps = []
    token_dict = {}
    for key in timestamps:
        times = timestamps[key]
        a = times  < future_time
        b = times  > current_time 
        token_times = times[a & b]
        if len(a) > 0:
            
            indice = np.where(times[len(a)-1] == times)[0][0]
            if indice < len(times) -1:
                is_on_staric_step = np.isclose(future_time, times[indice+1], rtol=rtol, atol=atol,
                                           equal_nan=True)
                if is_on_staric_step:
                    static_steps.append(key)
        if len(token_times) != 0 :
            token_dict[key] = token_times
    return token_dict, static_steps

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
    parser.add_argument("--restart_mother_folder", default='.', help="indicate where calc files located", type=str)
    parser.add_argument("--calc_dynamic_steps", default='0', help="whether preform dynamic steps between static steps ", type=int)
    parser.add_argument("--rdm_query_config_file", default='default_config.pickle', help="config file of dynamic steps: rdm ", type=str)




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
    is_restart = args.restart
    restart_mother_folder = args.restart_mother_folder
    is_calc_dynamic_steps = args.calc_dynamic_steps
    rdm_query_config_file = args.rdm_query_config_file
    # parm translate
    s_reno = s
    alpha_reno = 4*alpha # tranlate from wang1 to PRL

    # set log
    logger = package_logger

    job_name = f"traj_s{s:.2f}_alpha{alpha:.2f}_Omega{Omega}_omega_c{omega_c}_nmodes{nmodes}_bond_dims{bond_dims}_td_method_{td_method}_rho_type_{rho_type}"  ####################

    dump_dir = job_name
    os.makedirs(job_name, exist_ok=True)
    log.register_file_output(os.path.join(dump_dir, f'{job_name}.log'), mode="w")
    if is_restart:
        restart_folder = os.path.join(restart_mother_folder, job_name)
    # use old settings from example
    # nmodes = 1000
    Ms = bond_dims
    # upper_limit = 30

    Delta = Omega * 0.5
    eps = 0
    if is_calc_dynamic_steps :
        timestamps = read_query_config(rdm_query_config_file)
        logger.info('calc dynamic steps')
        logger.info(timestamps)

    # call a sdf
    sdf = rho_ohmic(alpha_reno, omega_c_reno, s_reno, rho_type)


    # wang1
    w, c2 = sdf.Wang1(nmodes)
    c = np.sqrt(c2)
    w_eff = w[w< omega_c_reno] # which are not noise
    logger.info(f"w:{w}")
    logger.info(f"c:{c}")
    with open(os.path.join(dump_dir, f'sdf_wang1_omega.pickle'), 'wb') as f:
        pickle.dump(w, f)
    with open(os.path.join(dump_dir, f'sdf_wang1_c.pickle'), 'wb') as f:
        pickle.dump(c, f)
    # reno ?
    reno = sdf.reno(w[-1])
    logger.info(f"renormalization constant: {reno}")
    Delta *= reno

    ham_terms = []



    # h_s
    # sigma_z sigma_x
    ham_terms.extend([Op("sigma_z", "spin", factor=eps, qn=0),
            Op("sigma_x", "spin", factor=Delta, qn=0)])
    # ham_terms.append(Op("sigma_x","spin",factor=Delta, qn=0))

    # boson energy
    for imode in range(nmodes):
        op1 = Op(r"p^2", f"v_{imode}", factor=0.5, qn=0)
        op2 = Op(r"x^2", f"v_{imode}", factor=0.5*w[imode]**2, qn=0)
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

    # restart dynamic
    if is_restart:
        points = sorted([ s for s in os.listdir(restart_folder) if s.endswith('step_ttns.npz')], reverse=False)
        if len(points) == 0:
            ttns.load(basis_tree, fname=os.path.join(restart_folder, points[0]))
            logger.info(f'restart from {points[0]}')
        else:
            logger.info('can not find restart file run from step 0 ')

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

    dt: float = 0.1/Omega # 
    logger.info(f'dt: {dt}')
    expectations: List[Union[float, complex]] = []
    entropy_1sites_traj: List[Dict[Union[int, List], float]] = []
    mutual_info_traj =[]
    # logger.info("ttns.basis.dof_list")
    # logger.info(ttns.basis.dof_list)
    # logger.info("ttns.basis.dof2idx")
    # logger.info(ttns.basis.dof2idx)
    calc_flag = False

    # step 0 is not init state of dynamic
    for i in range(nsteps):
        logger.info(f'proceeding step {i}')
        dump_file = os.path.join(dump_dir, f'{job_name}_{i}_step_ttns.npz')
        if is_restart:
            if dump_file == points[0]:
                calc_flag = True
        else:
            calc_flag = True
        if not calc_flag:
            continue
        ttns = ttns.evolve(ttno, dt)
        # store restart file
        
        ttns.dump(dump_file)

        if f'{job_name}_{i-1}_step_ttns.npz' in os.listdir(dump_dir):
            os.remove(os.path.join(dump_dir, f'{job_name}_{i-1}_step_ttns.npz'))

        # prop calc
        z: Union[float, complex] = ttns.expectation(exp_z)
    
        if is_calc_1sites_entropy  : 
            # vdof
            dofs = [f'v_{i}' for i in range(w_eff.shape[0]) if (i+1)%5 ==0 ]
            dofs.append('spin')
            logger.info(f'dofs:, {dofs}')
            entropy_1sites: Dict[Union[int, List], float] = ttns.calc_1dof_entropy(dofs)

            with open(os.path.join(dump_dir, f'{i:04}_step_entropy_1site.pickle'), 'wb') as f:
                pickle.dump(entropy_1sites, f)
            entropy_1sites_traj.append(entropy_1sites)
            logger.info(f'step {i} entropy_1sites: {entropy_1sites}')

        if is_calc_mutual_info:
            # spin-vdof
            dof1 ='spin'
            dofs = [(dof1, f'v_{i}')for i in range(w_eff.shape[0]) if (i+1)%5==0 ]
            rdm_2dof = ttns.calc_2dof_rdm(dofs)
            mutual_infos, entropy_tuple = ttns.calc_2dof_mutual_info(dofs, rdm_2dof)
            with open(os.path.join(dump_dir, f'{i:04}_step_mutual_infos.pickle'), 'wb') as f:
                pickle.dump(mutual_infos, f) 

            with open(os.path.join(dump_dir, f'{i:04}_step_entropy_int.pickle'), 'wb') as f:
                pickle.dump(mutual_infos, f) 
            # with open(os.path.join(dump_dir, f'{i:04}_step_rdm_2dofs.pickle'), 'wb') as f:
            #     pickle.dump(rdm_2dof, f) 
            mutual_info_traj.append(mutual_infos)
            logger.info(f'step {i} mutual_infos: {mutual_infos}')  

        expectations.append(z)
        logger.info(f'step {i} z: {z}')

        # dynamic step
        if is_calc_dynamic_steps :
            current_time = i * dt
            future_time = current_time + dt
            token_dict, static_step_dofs = check_onestep(timestamps, current_time, future_time, rtol=1e-7, atol=0)

            if len(token_dict) != 0 :
                ttns_dynamic = deepcopy(ttns)
                logger.info(f'wokring on dynamic step {i}')
                # do not sim by time 
                # less than dt just evolve  from current time
                dynamic_rdm_dof_dict = {}
                for key in token_dict:
                    # enum all dofs 
                    if key not in list(dynamic_rdm_dof_dict.keys()):
                        dynamic_rdm_dof_dict[key] = []
                    # enum all time in one dof
                    time_array = token_dict[key]
                    for i_time in range(len(time_array)):
                        time = time_array[i_time]
                        evolve_time = time - current_time
                        static_ttns = ttns.evolve(ttno, evolve_time)
                        if isinstance(key, str):
                            rdm_dof_dict = static_ttns.calc_1dof_rdm(key)
                        elif isinstance(key, tuple):
                            rdm_dof_dict = static_ttns.calc_2dof_rdm(key)
                        dynamic_rdm_dof_dict[key].append((time, rdm_dof_dict))

                with open(os.path.join(dump_dir, f'{i:04}_step_dynamic_rdm.pickle'), 'wb') as f:
                    pickle.dump(dynamic_rdm_dof_dict, f)

            if len(static_step_dofs) != 0:
                # rdm_dof_dict = {}
                logger.info(f'wokring on static step {i}')
                static_ttns = ttns.evolve(ttno, dt)
                if isinstance(static_step_dofs[0], str):
                    rdm_dof_dict = static_ttns.calc_1dof_rdm(static_step_dofs)
                elif isinstance(static_step_dofs[0], tuple):
                    rdm_dof_dict = static_ttns.calc_2dof_rdm(static_step_dofs)
                with open(os.path.join(dump_dir, f'{i:04}_step_static_rdm.pickle'), 'wb') as f:
                    pickle.dump(rdm_dof_dict, f)


    with open(os.path.join(dump_dir, f'expectations.pickle'), 'wb') as f:
        pickle.dump(expectations, f)
    # if is_calc_1sites_entropy:
    #     with open(os.path.join(dump_dir, f'entropy_1sites_traj.pickle'), 'wb') as f:
    #         pickle.dump(entropy_1sites_traj, f)
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