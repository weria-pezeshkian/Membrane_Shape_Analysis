from __future__ import annotations

from ..core.fourier_sft import SFT
from scipy.special import jv #move elsewhere potentially, but these two are needed in the calibrate function
import numpy as np

def calibrate(sft: SFT, radius: float, out_path: str) -> None:
    """Compute calibrated membrane material parameters from `sft` and write them to `out_path`.

    Not yet implemented - this is where the physics (e.g. kappa/sigma
    extraction from the Anm fluctuation spectrum) plugs in. See TODO.md's
    "Regularized Anm must never feed kappa/sigma calibration" entry for the
    constraint any such implementation must respect (unregularized Anm only).
    """
    

    #need:
    #A_mn, q_mn for all frames ->flatten to A_i, q_i_x, q_i_y; #A_mn_original=sft.A_mn; #q_mn_original=sft.q_mn

    frames = sft.A_mn.shape[0]
    A_i = sft.A_mn.reshape(frames, -1)
    q_i = sft.q_mn.reshape(frames, 2, -1)#qx_i,qy_i = q[:, 0], q[:, 1]

    #CHECK that the A_mn, q_mn stem from a trajectory that was centered!

    # Magnitude |q_i| for every frame and Fourier mode.
    q_abs_i = np.linalg.norm(q_i, axis=1)
    
    #now we need the integrals over the fourier functions F(r,q_i_x,q_i_y) at given q_i_x/y over a region of real space r (a matrix of functions)
    #(the protein region, in radius R_pro from center of box)
    ###For the integrals there are closed-form (approximations) via bessel functions, as suggested in COM.pdf 
    # - we might want to implement these directly for the integrals instead of numerical integration of the functions here!

    # Integral 1: Integral_P (F(q_i*r dA)=Integral_P([cos(q_i·r) + sin(q_i·r)] dA)=2*pi*radius*J_1(|q|R)/|q|; limit |q|->0=pi*R^2
    integrals_1 = np.empty_like(q_abs_i)
    
    zero_mode = np.isclose(q_abs_i, 0.0) #for the mode |q|=0; default relative tolerance of np. is 10^-5
    integrals_1[zero_mode] = np.pi * radius**2

    integrals_1[~zero_mode] = (2.0* np.pi * radius/ q_abs_i[~zero_mode]* jv(1, q_abs_i[~zero_mode] * radius)) #jv=bessel func. from scipy.special

    # Integral 2: Integral_P ([F(q_i*r)xF(q_j*r)] dA)=Integral_P(cos([q_i-q_j]·r) dA)=2*pi*radius*J_1(|q_i-q_j|R)/|q_i-q_j|; limit |q_i-q_j|->0=pi*R^2
    integrals_2 = #np.empty_like(q_abs_i)
    
    zero_mode2 = #np.isclose(q_abs_i, 0.0) #for the mode |q|=0; default relative tolerance of np. is 10^-5
    integrals_2[zero_mode] = #np.pi * radius**2

    integrals_2[~zero_mode] =# (2.0* np.pi * radius/ q_abs_i[~zero_mode]* jv(1, q_abs_i[~zero_mode] * radius)) #jv=bessel func. from scipy.special

    


    #1. calculate the time averages <A_i> and <A_i*A_j>-<A_i>*<A_j> (a vector and a matrix)
    avg_Ai=np.mean(A_i,axis=0)
    A_matrix=np.einsum("fi,fj->fij", A_i, A_i) #shape(frames, #A_modes, #A_modes)
    avg_outer = np.einsum("i,j->ij", avg_Ai, avg_Ai)  #shape (#A_modes,#A_modes)
    sigma_A=np.mean(A_matrix,axis=0) - avg_outer #shape (#A_modes,#A_modes)
    
    #2. for each frame: take the integrals over F(r,q_i_x, q_i_y) etc. to get the Matrix M and vector C
    #3. for each frame solve the system of equations: 
    #3. a) solve M*sigma(A)=I to get kappa, D_kappa, D_kappa_g
    #3. b) use a) and solve <A>=1/2 C*M^-1 +1/2 M^-1*C to get C_0
    #4. determine the averages of kappa, D_kappa, D_kappa_g, C_0 over all frames and save them as output


    raise NotImplementedError("CALM calibrate's physics is not yet implemented.")
