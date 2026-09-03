#TODo: #make sense of the output values & check units... for now, tested on CholeraToxin, rep1, frames 0-5000 of prod.xtc, they do NOT make sense
from __future__ import annotations

from ..core.fourier_sft import SFT
from scipy.special import jv
from scipy.optimize import least_squares
import numpy as np

def calibrate(sft: SFT, radius: float, frame_start: int, frame_end: int, out_path: str,surface: str = "top") -> None:
    """Compute calibrated membrane material parameters from `sft` and write them to `out_path`.
    radius=choose radius of area under protein; 
    frame_start/end: chose over which frames to average q_i-integrals; 
    outpath: where to save results;

    This is where the physics (extraction of kappa, delta_kappa, delta_kappa_g, Co
    from the Anm fluctuation spectrum) plugs in. Any such
    implementation must assert `sft.regularized in (False, None)` first:
    Tikhonov regularization biases Anm toward zero in proportion to
    curvature, which would circularly contaminate a fluctuation-spectrum
    fit built from it.
    NOTE: smallest wavelength for fourier fit cutoff should be close to mem. thickness, not smaller!
    """
    assert sft.regularized in (False, None), "Calibration requires an unregularized SFT."
    
    #A_mn, q_mn for all frames ->flatten to A_i, q_i_x, q_i_y; #A_mn_original=sft.A_mn; #q_mn_original=sft.q_mn
    #sft.A_mn.shape = (5006, 3, 7, 7) sft.q_mn.shape = (5006, 2, 7, 7); 2=q_x, q_y; 3=Amn_top, middle, bottom

    surface_idx = {"top": 0,"middle": 1,"bottom": 2}[surface] #to choose which surface to use for calibration analysis (which part of Amn to use)
    frames = sft.A_mn.shape[0]
    A_i = sft.A_mn[:, surface_idx, :, :].reshape(frames, -1)
    q_i_all = sft.q_mn.reshape(frames, 2, -1)
    q_i=q_i_all[frame_start:frame_end]
    n_frames = q_i.shape[0] #number of user chosen frames for integral calculations (see 2.)

    #testing
    #print("q_i_shape=",q_i.shape)
    #print("A_i_shape=",A_i.shape, "A_mn_shape=",sft.A_mn.shape)
    #exit()

    #CHECK that the A_mn, q_mn come from a trajectory that was centered!

    #1. calculate the time averages <A_i> and <A_i*A_j>-<A_i>*<A_j> (a vector and a matrix) OVER ALL FRAMES
    avg_Ai=np.mean(A_i,axis=0)
    A_matrix=np.einsum("fi,fj->fij", A_i, A_i) #shape(frames, #A_modes, #A_modes)
    avg_outer = np.einsum("i,j->ij", avg_Ai, avg_Ai)  #shape (#A_modes,#A_modes)
    sigma_A=np.mean(A_matrix,axis=0) - avg_outer #shape (#A_modes,#A_modes)

    #2. take the integrals over F(r,q_i_x, q_i_y) etc. to get the Matrix M and vector C;
    #  ONLY FOR CHOSEN AMOUNT OF FRAMES and take mean over those, since box can vary - in theory we assume q_i are the same for all frames

    # Magnitude |q_i| for every frame and Fourier mode.
    q_abs_i = np.linalg.norm(q_i, axis=1)
    Nmodes = q_abs_i.shape[1] #shape[0]=n_frames
    L_x, L_y=sft.dimensions[frame_start:frame_end, 0], sft.dimensions[frame_start:frame_end, 1]


    # Integral 1: VECTOR;Integral_P (F(q_i*r dA)=Integral_P([cos(q_i·r) + sin(q_i·r)] dA)=2*pi*radius*J_1(|q|R)/|q|; limit |q|->0=pi*R^2
    
    #needed vector: C_i=(kappa+delta_kappa)*C_0*2*pi*radius*|q_i|*J_1(|q_i|*R)=factor*|q_i|*^2*integral
    integrals_1 = (2.0* np.pi * radius* q_abs_i* jv(1, q_abs_i * radius)) #jv=bessel func. from scipy.special
    C_vec_avg=np.mean(integrals_1,axis=0)

    # Integral 2: MATRIX;Integral_P ([F(q_i*r)xF(q_j*r)] dA)=Integral_P(cos([q_i-q_j]·r) dA)=2*pi*radius*J_1(|q_i-q_j|R)/|q_i-q_j|; limit |q_i-q_j|->0=pi*R^2
    qx = q_i[:, 0]
    qy = q_i[:, 1]
    dqx = qx[:, :, None] - qx[:, None, :]
    dqy = qy[:, :, None] - qy[:, None, :]
    dq_abs = np.sqrt(dqx**2 + dqy**2) #MATRIX dq_abs(i,j)=sqrt((qx_i-qx_j)^2+(qy_i-qy_j)^2)

    integrals_2 = np.empty_like(dq_abs)
    zero_mode = np.isclose(dq_abs, 0.0)
    integrals_2[zero_mode] = np.pi * radius**2
    integrals_2[~zero_mode] = (2.0*np.pi* radius* jv(1, dq_abs[~zero_mode] * radius)/ dq_abs[~zero_mode])

    #M=kappa*H_M + delta_kappa*H_P - 2*delta_kappa_g* K_P
    H_M = np.zeros((n_frames, Nmodes, Nmodes)) #diagonal matrix; L_x*L_y*|q_i|^2 * |q_j|^2 *delta_ij
    idx = np.arange(Nmodes)
    H_M[:, idx, idx] =  L_x[:, None]*L_y[:, None]*q_abs_i**4 #reshaping of Lxy needed for correct multiframe*multimode multiplication
    H_M_avg=np.mean(H_M,axis=0)

    H_P=q_abs_i[:, :, None]**2* q_abs_i[:, None, :]**2* integrals_2 #matrix |q_i|^2 * |q_j|^2* 2*pi*radius*J_1(|q_i-q_j|R)/|q_i-q_j|
    H_P_avg=np.mean(H_P,axis=0)

    K_factor = (qx[:, :, None]**2 * qy[:, None, :]**2- qx[:, :, None] * qx[:, None, :] * qy[:, :, None] * qy[:, None, :])
    K_P = K_factor * integrals_2 #matrix (q_i_x^2*q_j_y^2 − q_i_x*q_j_x*q_i_y*q_j_y) *2*pi*radius*J_1(|q_i-q_j|R)/|q_i-q_j|
    #print("shape K_factor:", K_factor.shape, ",integrals_2:",integrals_2.shape, ",K_P:",K_P.shape)
    K_P_avg=np.mean(K_P,axis=0)
    #print("K_P_avg shape:",K_P_avg.shape)



    #3. for each frame solve the system of equations: 
    #3. a) solve M*sigma(A)=I to get kappa, delta_kappa, delta_kappa_g
    def residuals_1(params):
        kappa, delta_kappa, delta_kappa_g = params
        M = ( kappa * H_M_avg+ delta_kappa * H_P_avg - 2.0 * delta_kappa_g * K_P_avg)
        return (M @ sigma_A - np.eye(Nmodes)).ravel()
   
    result = least_squares(residuals_1,x0=(20.0, 0.0, 0.0),bounds=( #default of 20kT for kappa here, CHECK whether this is correct units
         (0.0, -50.0, -50.0),   # lower bounds
        (50.0, 50.0, 50.0)    # upper bounds
    )) 
    if not result.success:
        raise RuntimeError(result.message)
   
    kappa, delta_kappa, delta_kappa_g = result.x


    #3. b) use a) and solve  for C_o: <A>=(kappa+delta_kappa)*C_o* M^-1 *C_vec_avg <=> M<A>=(kappa+delta_kappa)*C_o*C_vec_avg
    M_matrix = ( kappa * H_M_avg+ delta_kappa * H_P_avg - 2.0 * delta_kappa_g * K_P_avg)

    def residuals_2(params):
        c_zero = params[0]
        return (M_matrix @ avg_Ai- (kappa + delta_kappa) * c_zero * C_vec_avg).ravel()

    result2 = least_squares(residuals_2,x0=(0.0)) #default of no spont. curvature
    if not result2.success:
        raise RuntimeError(result2.message)
   
    c_zero= result2.x[0]

    #4. save kappa, D_kappa, D_kappa_g, C_0 as output
    parameters = {
    "kappa": kappa,
    "delta_kappa": delta_kappa,
    "delta_kappa_g": delta_kappa_g,
    "C_0": c_zero}
    if not out_path.endswith(".npy"):
        out_path += ".npy"
    np.save(out_path, parameters, allow_pickle=True)

    # Save as text file too
    txt_path = out_path[:-4] + ".txt"
    with open(txt_path, "w") as f:
        f.write(f"kappa = {kappa} [kT]\n")
        f.write(f"delta_kappa = {delta_kappa} [kT]\n")
        f.write(f"delta_kappa_g = {delta_kappa_g} [kT]\n")
        f.write(f"C_0 = {c_zero} [1/A]\n")

    #And print
    print(f"kappa = {kappa} [kT],delta_kappa = {delta_kappa} [kT],delta_kappa_g = {delta_kappa_g} [kT], C_0 = {c_zero} [1/A]")

    print(f"Done(preliminary version!), check your results under {txt_path}")