import numpy as np
from scipy import optimize
from scipy.constants import mu_0, epsilon_0
from scipy import fftpack
from scipy import sparse
from scipy.special import factorial
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d, CubicSpline,splrep, BSpline
from scipy.sparse import csr_matrix, csc_matrix
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import lu_factor, lu_solve
from scipy import signal
import empymod
import discretize
import  os
eps= np.finfo(float).eps

class TikonovInversion:
    def __init__(self, G_f, Wd, alphax=1.,Wx=None,
        alphas=1., Ws=None, m_ref=None,Proj_m=None,m_fix=None,
        sparse_matrix=False
        ):  
        self.G_f = G_f
        self.Wd = Wd
        self.Wx = Wx
        self.Ws = Ws
        self.nD = G_f.shape[0]
        self.nP = G_f.shape[1]
        self.alphax = alphax
        self.Proj_m = Proj_m  
        self.m_fix = m_fix
        if Proj_m is not None:
            assert Proj_m.shape[0] == self.nP
            self.nM = Proj_m.shape[1]
        else:
            self.Proj_m = np.eye(self.nP)
            self.nM = self.nP
            self.m_fix = np.zeros(self.nP)
        self.alphas = alphas
        self.m_ref=m_ref

        self.sparse_matrix = sparse_matrix
    
    def get_Wx(self):
        nP = self.nP
        Wx = np.zeros((nP-1, nP))
        element = np.ones(nP-1)
        Wx[:,:-1] = np.diag(element)
        Wx[:,1:] += np.diag(-element)
        self.Wx = Wx
        return Wx
    
    def get_Ws(self):
        nM = self.nM
        Ws = np.eye(nM)
        self.Ws=  Ws
        return Ws

    def recover_model(self, dobs, beta, sparse_matrix=False):
        # This is for the mapping 
        G_f = self.G_f
        Wd = self.Wd
        alphax = self.alphax
        alphas = self.alphas
        Wx = self.Wx
        Ws = self.Ws
        m_ref= self.m_ref
        Proj_m = self.Proj_m
        m_fix= self.m_fix
        sparse_matrix = self.sparse_matrix
        
        left = Proj_m.T @G_f.T @ Wd.T @ Wd @ G_f@Proj_m
        left += beta * alphax * (Proj_m.T @Wx.T @ Wx@Proj_m) 
        if m_ref is not None:
            left += beta * alphas * (Ws.T @ Ws)
        if sparse_matrix:
            left = csr_matrix(left)
        right =   G_f.T @ Wd.T @Wd@ dobs@Proj_m
        right += -m_fix.T@G_f.T@Wd.T@Wd@G_f@Proj_m
        right+= -beta*alphax* m_fix.T@Wx.T@Wx@Proj_m
        if m_ref is not None:
            right+= beta*alphas*m_ref.T@Ws.T@Ws
        m_rec = np.linalg.solve(left, right)
        #filt_curr = spsolve(left, right)
        rd = Wd@(G_f@Proj_m@m_rec-dobs)
        rmx = alphax*Wx@Proj_m@m_rec
        if m_ref is not None:
            rms = alphas*Ws@(m_rec-m_ref)

        phid = 0.5 * np.dot(rd, rd)
        phim = 0.5 * np.dot(rmx,rmx)
        if m_ref is not None:
            phim+=0.5 * np.dot(rms,rms)
        p_rec = m_fix + Proj_m@m_rec
        return p_rec, phid, phim
    
    def tikonov_inversion(self,beta_values, dobs):
        n_beta = len(beta_values)
        nP= self.nP

        mrec_tik = np.zeros(nP, n_beta)  # np.nan * np.ones(shape)
        phid_tik = np.zeros(n_beta)
        phim_tik = np.zeros(n_beta) 
        for i, beta in enumerate(beta_values): 
            mrec_tik[:, i], phid_tik[i], phim_tik[i] = self.recover_model(
            dobs=dobs, beta=beta)
        return mrec_tik, phid_tik, phim_tik

    
    def estimate_beta_range(self, num=20, eig_tol=1e-12):
        G_f = self.G_f
        alphax=self.alphax
        alphas=self.alphas
       
        Wd = self.Wd
        Wx = self.Wx
        Ws= self.Ws
        Proj_m = self.Proj_m  # Use `Proj_m` to map the model space

        # Effective data misfit term with projection matrix
        A_data = Proj_m.T @ G_f.T @ Wd.T @ Wd @ G_f @ Proj_m
        eig_data = np.linalg.eigvalsh(A_data)
        
        # Effective regularization term with projection matrix
        A_reg = alphax* Proj_m.T @ Wx.T @ Wx @ Proj_m
        if Ws is not None:
            A_reg += alphas * (Ws.T @ Ws)
        eig_reg = np.linalg.eigvalsh(A_reg)
        
        # Ensure numerical stability (avoid dividing by zero)
        eig_data = eig_data[eig_data > eig_tol]
        eig_reg = eig_reg[eig_reg > eig_tol]

        # Use the ratio of eigenvalues to set beta range
        beta_min = np.min(eig_data) / np.max(eig_reg)
        beta_max = np.max(eig_data) / np.min(eig_reg)
        
        # Generate 20 logarithmically spaced beta values
        beta_values = np.logspace(np.log10(beta_min), np.log10(beta_max), num=num)
        return beta_values

class projection_convex_set:
    def __init__(self,maxiter=100, tol=1e-2,
        lower_bound=None, upper_bound=None, a=None, b=None):
        self.maxiter = maxiter
        self.tol = tol
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.a = a
        self.b = b

    def get_param(self, param, default):
        return param if param is not None else default
        
    def projection_halfspace(self, a, x, b):
        a = self.get_param(a, self.a)
        b = self.get_param(b, self.b)
        projected_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a)) if np.dot(a, x) > b else x
        # Ensure scalar output if input x is scalar
        if np.isscalar(x):
            return float(projected_x)
        return projected_x

    def projection_plane(self, a, x, b):
        a = self.get_param(a, self.a)
        b = self.get_param(b, self.b)
        projected_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a))
        # Ensure scalar output if input x is scalar
        if np.isscalar(x):
            return float(projected_x)
        return projected_x

    def clip_model(self, x, lower_bound=None, upper_bound=None):
        lower_bound = self.get_param(lower_bound, self.lower_bound)
        upper_bound = self.get_param(upper_bound, self.upper_bound)
        clipped_x = np.clip(x, self.lower_bound, self.upper_bound)
        return clipped_x

    def proj_c(self,x, maxiter=100, tol=1e-2):
        "Project model vector to convex set defined by bound information"
        x_c_0 = x.copy()
        x_c_1 = np.zerps_like(x)
        maxiter = self.get_param(maxiter, self.maxiter)
        tol = self.tol
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        a = self.a
        b = self.b
        for i in range(maxiter):
            x_c_1 = self.clip_model(x=x_c_0,lower_bound=lower_bound, upper_bound=upper_bound)
            x_c_1 = self.projection_plane(a=a, x=x_c_1, b=b)
            if np.linalg.norm(x_c_1 - x_c_0) < tol:
                break
            x_c_0 = x_c_1
        return x_c_1

class empymod_IPinv:

    def __init__(self, model_base, nlayer,
        m_ref=None, nD=0, nlayer_fix=0, Prj_m=None, m_fix=None,
        nM_r = None, nM_m = None, nM_t= None, nM_c=None,
        recw=None,resmin=1e-3 , resmax=1e6, chgmin=1e-3, chgmax=0.9,
        taumin=1e-6, taumax=1e-1, cmin= 0.4, cmax=0.9,
        Wd = None, Ws=None,Ws_threshold=1e-12, Wx=None, alphax=None, alphas=None,
        cut_off=None,filt_curr = None,  window_mat = None,
        ):
        self.model_base = model_base
        self.nlayer = int(nlayer)
        self.nlayer_fix = int(nlayer_fix)
        self.nP = 4*(nlayer + nlayer_fix)
        self.m_ref = m_ref
        self.Prj_m = Prj_m  
        self.m_fix = m_fix
        self.recw = recw
        if Prj_m is not None:
            assert Prj_m.shape[0] == self.nP
            self.nM = Prj_m.shape[1]
        else:
            self.Prj_m = np.eye(self.nP)
            self.nM = self.nP
            self.m_fix = np.zeros(self.nP) 
            self.nM_r = nlayer
            self.nM_m = nlayer
            self.nM_t = nlayer
            self.nM_c = nlayer
        self.nD = nD
        self.resmin = resmin
        self.resmax = resmax
        self.chgmin = chgmin
        self.chgmax = chgmax
        self.taumin = taumin
        self.taumax = taumax
        self.cmin = cmin
        self.cmax = cmax
        self.Wd = Wd
        self.Ws = Ws
        self.Ws_threshold = Ws_threshold
        self.Wx = Wx
        self.alphax = alphax
        self.alphas = alphas
        self.cut_off = cut_off
        self.filt_curr = filt_curr
        self.window_mat = window_mat

    def get_param(self, param, default):
        return param if param is not None else default
        
    def fix_sea_basement(self, res_sea, res_base, 
                chg_sea, chg_base, tau_sea, tau_base, c_sea, c_base):
        ## return and set mapping for fixigin sea and basement resistivity
        ## Assert there are no fix ing at this stage
        nlayer = self.nlayer
        nlayer_fix=2
        nlayer_sum = nlayer+nlayer_fix
        Prj_m_A = np.block([
            [np.zeros(nlayer)], # sea water
            [np.eye(nlayer)], # layers
            [np.zeros(nlayer)], # basement
        ])
        Prj_m=np.block([
        [Prj_m_A, np.zeros((nlayer_sum, 3*nlayer))], # Resistivity
        [np.zeros((nlayer_sum,  nlayer)), Prj_m_A, np.zeros((nlayer_sum, 2*nlayer))], # Chargeability
        [np.zeros((nlayer_sum,2*nlayer)), Prj_m_A, np.zeros((nlayer_sum, nlayer))], # Time constant
        [np.zeros((nlayer_sum,3*nlayer)), Prj_m_A], # Exponent C
        ])
        m_fix = np.r_[ 
        np.log(res_sea), np.zeros(nlayer), np.log(res_base), # Resistivity
        chg_sea, np.zeros(nlayer), chg_base, # Chargeability
        np.log(tau_sea),np.zeros(nlayer), np.log(tau_base), # Time constant
        c_sea,np.zeros(nlayer),c_base # Exponent C
        ]
        assert len(m_fix) == 4*nlayer_sum
        self.nlayer_fix = nlayer_fix
        self.Prj_m = Prj_m
        self.m_fix = m_fix
        self.nP= Prj_m.shape[0]
        self.nM= Prj_m.shape[1]
        assert self.nP == 4*(nlayer+nlayer_fix)
        assert self.nM == 4*nlayer
        return Prj_m, m_fix

    def fix_sea_tau_c(self,
             res_sea, chg_sea, tau_sea, c_sea):
        ## return and set mapping for fixigin sea and basement resistivity
        ## Assert there are no fix ing at this stage
        nlayer = self.nlayer
        nlayer_fix=1
        nlayer_sum = nlayer+nlayer_fix
        Prj_m_A = np.block([ # res chg
            [np.zeros(nlayer)], # sea water
            [np.eye(nlayer)], # layers
        ])
        Prj_m_b =np.block([ # tau c
            [0], # sea water
            [np.ones(nlayer).reshape(-1,1)], # layers
        ])
        Prj_m=np.block([
        [Prj_m_A, np.zeros((nlayer_sum, nlayer+2))], # Resistivity
        [np.zeros((nlayer_sum,  nlayer)), Prj_m_A, np.zeros((nlayer_sum, 2))], # Chargeability
        [np.zeros((nlayer_sum,2*nlayer)), Prj_m_b, np.zeros(nlayer_sum).reshape(-1,1)], # Time constant
        [np.zeros((nlayer_sum,2*nlayer)), np.zeros(nlayer_sum).reshape(-1,1), Prj_m_b], # Exponent C
        ])
        m_fix = np.r_[ 
        np.log(res_sea), np.zeros(nlayer), # Resistivity
        chg_sea, np.zeros(nlayer), # Chargeability
        np.log(tau_sea),np.zeros(nlayer), # Time constant
        c_sea,np.zeros(nlayer) # Exponent C
        ]
        assert len(m_fix) == 4*nlayer_sum
        self.nlayer_fix = nlayer_fix
        self.Prj_m = Prj_m
        self.m_fix = m_fix
        self.nP= Prj_m.shape[0]
        self.nM= Prj_m.shape[1]
        self.nM_r = nlayer
        self.nM_m = nlayer
        self.nM_t = 0
        self.nM_c = 0
        assert self.nP == 4*(nlayer_sum)
        assert self.nM == 2*nlayer
        return Prj_m, m_fix

    def get_Wx_sea_basement(self):
        nlayer = self.nlayer
        nlayer_fix=2
        nlayer_sum = nlayer+nlayer_fix
        nM = self.nM
        nP = self.nP
        Wx = np.zeros((nM,nP))
        if nlayer == 1:
            print("No smoothness for one layer model")
            self.Wx = Wx
            return Wx
        Wx_block = np.zeros((nlayer-1, nlayer_sum))
        Wx_block[:,1:-2] = np.eye(nlayer-1)
        Wx_block[:,2:-1] -= np.eye(nlayer-1)
        Wx=np.block([
        [Wx_block, np.zeros((nlayer-1, nlayer_sum*3))], # Resistivity
        [np.zeros((nlayer-1, nlayer_sum*1)), Wx_block, np.zeros((nlayer-1, nlayer_sum*2))], # Chargeability
        [np.zeros((nlayer-1, nlayer_sum*2)), Wx_block, np.zeros((nlayer-1, nlayer_sum*1))], # Time constant
        [np.zeros((nlayer-1, nlayer_sum*3)), Wx_block], # Exponent C
        ])
        self.Wx = Wx
        return Wx

    def fix_sea_one_tau_c(self,
             res_sea, chg_sea, tau_sea, c_sea):
        ## return and set mapping for fixigin sea and basement resistivity
        ## Assert there are no fix ing at this stage
        nlayer = self.nlayer
        nlayer_fix=1
        nlayer_sum = nlayer+nlayer_fix
        Prj_m_A = np.block([ # res chg
            [np.zeros(nlayer)], # sea water
            [np.eye(nlayer)], # layers
        ])
        Prj_m_b =np.block([ # tau c
            [0], # sea water
            [np.ones(nlayer).reshape(-1,1)], # layers
        ])
        Prj_m=np.block([
        [Prj_m_A, np.zeros((nlayer_sum, nlayer+2))], # Resistivity
        [np.zeros((nlayer_sum,  nlayer)), Prj_m_A, np.zeros((nlayer_sum, 2))], # Chargeability
        [np.zeros((nlayer_sum,2*nlayer)), Prj_m_b, np.zeros(nlayer_sum).reshape(-1,1)], # Time constant
        [np.zeros((nlayer_sum,2*nlayer)), np.zeros(nlayer_sum).reshape(-1,1), Prj_m_b], # Exponent C
        ])
        m_fix = np.r_[ 
        np.log(res_sea), np.zeros(nlayer), # Resistivity
        chg_sea, np.zeros(nlayer), # Chargeability
        np.log(tau_sea),np.zeros(nlayer), # Time constant
        c_sea,np.zeros(nlayer) # Exponent C
        ]
        assert len(m_fix) == 4*nlayer_sum
        self.nlayer_fix = nlayer_fix
        self.Prj_m = Prj_m
        self.m_fix = m_fix
        self.nP= Prj_m.shape[0]
        self.nM= Prj_m.shape[1]
        self.nM_r = nlayer
        self.nM_m = nlayer
        self.nM_t = 1
        self.nM_c = 1
        assert self.nP == 4*(nlayer_sum)
        assert self.nM == 2*nlayer + 2
        return Prj_m, m_fix

    def get_Wx_rm(self):
        nlayer = self.nlayer
        nM_r = self.nM_r
        nM_m = self.nM_m
        nM_t = self.nM_t
        nM_c = self.nM_c
        depth = self.model_base["depth"]
        depth= np.r_[depth,2*depth[-1]-depth[-2]]
        x = (depth[:-1] + depth[1:]) / 2


        if nlayer == 1:
            print("No smoothness for one layer model")
            nM = self.nM
            nP = self.nP
            Wx = np.zeros((nM,nP))
            self.Wx = Wx
            return Wx
        delta_x = np.diff(x)
        elm1 = 1/delta_x
        elm2 = np.sqrt(delta_x)
        Wx_block = np.zeros((nlayer-1, nlayer))
        Wx_block[:,:-1] = -np.diag(elm2*elm1)
        Wx_block[:,1:] += np.diag(elm2*elm1)
        # Wx_block = np.diag(elm2) @Wx_block 

        # block_r = np.zeros((nM_r-1, nM_r))
        # block_r[:,:-1] = np.eye(nM_r)
        # block_r[:,1:] -= np.eye(nM_r)
        # block_m = np.zeros((nM_m-1, nM_m))
        # block_m[:,:-1] = np.eye(nM_m)
        # block_m[:,1:] -= np.eye(nM_m)
        Wx=np.block([
        [Wx_block, np.zeros((nM_r-1, nM_m + nM_t + nM_c ))], # Resistivity
        [np.zeros((nM_m-1, nM_r)), Wx_block, np.zeros((nM_m-1,  nM_t + nM_c ))], # Chargeability
        ])
        self.Wx = Wx
        return Wx
    
    def get_Ws_sea_one_tau_c(self):
        nlayer = self.nlayer
        nM_r = self.nM_r
        nM_m = self.nM_m
        nM_t = self.nM_t
        nM_c = self.nM_c
        depth = self.model_base["depth"]
        depth= np.r_[depth,2*depth[-1]-depth[-2]]
        delta_x = np.diff(depth)
        elm1 =  np.sqrt(delta_x)
        Ws_block1 = np.diag(elm1)
        Ws_block2=  np.r_[elm1.sum()]
        Ws = np.block([
            [Ws_block1, np.zeros((nlayer, nM_m)),np.zeros((nlayer,nM_t+nM_c))], # Resistivity
            [np.zeros((nlayer, nM_r)), Ws_block1,np.zeros((nlayer,nM_t+nM_c))], # Chargeabillity
            [np.zeros((1, nM_r+nM_m)), Ws_block2,np.zeros((1,nM_c))], # time_constant
            [np.zeros((1, nM_r+nM_m+nM_t)), Ws_block2], # time_constant
        ])
        self.Ws = Ws
        return Ws

    def fix_sea(self, res_sea, chg_sea, tau_sea, c_sea):
        ## return and set mapping for fixigin sea and basement resistivity
        ## Assert there are no fix ing at this stage
        nlayer = self.nlayer
        nlayer_fix=1
        nlayer_sum = nlayer+nlayer_fix
        Prj_m_A = np.block([
            [np.zeros(nlayer)], # sea water
            [np.eye(nlayer)], # layers
        ])
        Prj_m=np.block([
        [Prj_m_A, np.zeros((nlayer_sum, 3*nlayer))], # Resistivity
        [np.zeros((nlayer_sum,  nlayer)), Prj_m_A, np.zeros((nlayer_sum, 2*nlayer))], # Chargeability
        [np.zeros((nlayer_sum,2*nlayer)), Prj_m_A, np.zeros((nlayer_sum, nlayer))], # Time constant
        [np.zeros((nlayer_sum,3*nlayer)), Prj_m_A], # Exponent C
        ])
        m_fix = np.r_[ 
        np.log(res_sea), np.zeros(nlayer), # Resistivity
        chg_sea, np.zeros(nlayer), # Chargeability
        np.log(tau_sea),np.zeros(nlayer), # Time constant
        c_sea,np.zeros(nlayer)# Exponent C
        ]
        assert len(m_fix) == 4*nlayer_sum
        self.nlayer_fix = nlayer_fix
        self.Prj_m = Prj_m
        self.m_fix = m_fix
        self.nP= Prj_m.shape[0]
        self.nM= Prj_m.shape[1]
        self.nM_r = nlayer
        self.nM_m = nlayer
        self.nM_t = nlayer
        self.nM_c = nlayer
        assert self.nP == 4*(nlayer+nlayer_fix)
        assert self.nM == 4*nlayer
        return Prj_m, m_fix

    def get_Wx_sea(self):
        nlayer = self.nlayer
        nlayer_fix=1
        nlayer_sum = nlayer+nlayer_fix
        depth = self.model_base["depth"]
        depth= np.r_[depth,2*depth[-1]-depth[-2]]
        if nlayer == 1:
            nM = self.nM
            nP = self.nP
            Wx = np.zeros((nM,nP))
            print("No smoothness for one layer model")
            self.Wx = Wx
            return Wx
        Wx_block = np.zeros((nlayer-1, nlayer_sum))
        delta_x = np.diff(depth)
        elm1 = 1/delta_x[:-1]
        Wx_block[:,1:-1] = -np.diag(elm1)
        Wx_block[:,2:] += np.diag(elm1)
        elm2 = np.r_[0,np.sqrt(delta_x)]
        Wx_block = Wx_block@ np.diag(elm2)
        # Wx_block[:,1:-1] = np.eye(nlayer-1)
        # Wx_block[:,2:] -= np.eye(nlayer-1)
        Wx=np.block([
        [Wx_block, np.zeros((nlayer-1, nlayer_sum*3))], # Resistivity
        [np.zeros((nlayer-1, nlayer_sum*1)), Wx_block, np.zeros((nlayer-1, nlayer_sum*2))], # Chargeability
        [np.zeros((nlayer-1, nlayer_sum*2)), Wx_block, np.zeros((nlayer-1, nlayer_sum*1))], # Time constant
        [np.zeros((nlayer-1, nlayer_sum*3)), Wx_block], # Exponent C
        ])
        self.Wx = Wx
        return Wx

    def pelton_et_al(self, inp, p_dict):
        """ Pelton et al. (1978)."""

        # Compute complex resistivity from Pelton et al.
        iotc = np.outer(2j * np.pi * p_dict['freq'], inp['tau']) ** inp['c']
        rhoH = inp['rho_0'] * (1 - inp['m'] * (1 - 1 / (1 + iotc)))
        rhoV = rhoH * p_dict['aniso'] ** 2

        # Add electric permittivity contribution
        etaH = 1 / rhoH + 1j * p_dict['etaH'].imag
        etaV = 1 / rhoV + 1j * p_dict['etaV'].imag
        return etaH, etaV

    def get_ip_model(self, mvec):
        Prj_m = self.Prj_m
        m_fix = self.m_fix
        nlayer= self.nlayer
        nlayer_fix = self.nlayer_fix
        nlayer_sum = nlayer + nlayer_fix
        param = Prj_m @ mvec + m_fix
        res = np.exp(param[            :   nlayer_sum])
        m   =        param[  nlayer_sum: 2*nlayer_sum]
        tau = np.exp(param[2*nlayer_sum: 3*nlayer_sum])
        c   =        param[3*nlayer_sum: 4*nlayer_sum]
        pelton_model = {'res': res, 'rho_0': res, 'm': m,
                        'tau': tau, 'c': c, 'func_eta': self.pelton_et_al}
        return pelton_model


    def predicted_data(self, model_vector):
        cut_off = self.cut_off
        filt_curr = self.filt_curr
        window_mat = self.window_mat
        ip_model = self.get_ip_model(model_vector)
        data = empymod.bipole(res=ip_model, **self.model_base)
        if data.ndim == 3:
            # Sum over transmitter and receiver dimensions (axis 1 and axis 2)
            data=np.sum(data, axis=(1, 2))
        elif data.ndim == 2:
            # Sum over the transmitter dimension (axis 1)
            if self.recw is None:
                data= np.sum(data, axis=1)
            else:
                recw = self.recw
                data =(recw @ data.T).squeeze()

        self.nD = len(data)
        if cut_off is not None:
            times = self.model_base['freqtime']
            smp_freq = 1/(times[1]-times[0])
            data_LPF = self.apply_lowpass_filter(
                       data=data,cut_off=cut_off,smp_freq=smp_freq
                       )
            data = data_LPF
        if filt_curr is not None:
            data_curr = signal.convolve(data_LPF, filt_curr)[:len(data)]
            data = data_curr
        if window_mat is not None:
            data_window = window_mat @ data_curr
            self.nD = len(data_window)
            data = data_window
        return data
   
    def apply_lowpass_filter(self, data, cut_off,smp_freq, order=1):
        nyquist = 0.5 * smp_freq
        normal_cutoff = cut_off / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def projection_halfspace(self, a, x, b):
        projected_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a)) if np.dot(a, x) > b else x
        # Ensure scalar output if input x is scalar
        if np.isscalar(x):
            return float(projected_x)
        return projected_x

    def proj_c(self,mvec):
        "Project model vector to convex set defined by bound information"
        nlayer = self.nlayer
        a = np.r_[1]
        print(mvec)
        for j in range(nlayer):
            r_prj = mvec[j]
            m_prj = mvec[j+   nlayer]
            t_prj = mvec[j+ 2*nlayer]
            c_prj = mvec[j+ 3*nlayer]
            r_prj = float(self.projection_halfspace( a, r_prj,  np.log(self.resmax)))
            r_prj = float(self.projection_halfspace(-a, r_prj, -np.log(self.resmin)))
            m_prj = float(self.projection_halfspace( a, m_prj,  self.chgmax))
            m_prj = float(self.projection_halfspace(-a, m_prj, -self.chgmin))
            t_prj = float(self.projection_halfspace( a, t_prj,  np.log(self.taumax)))
            t_prj = float(self.projection_halfspace(-a, t_prj, -np.log(self.taumin)))
            c_prj = float(self.projection_halfspace( a, c_prj,  self.cmax))
            c_prj = float(self.projection_halfspace(-a, c_prj, -self.cmin))
            mvec[j         ] = r_prj
            mvec[j+  nlayer] = m_prj
            mvec[j+2*nlayer] = t_prj
            mvec[j+3*nlayer] = c_prj
        return mvec
  
    def clip_model(self, mvec):
        mvec_tmp = mvec.copy()
        # nlayer = self.nlayer
        index_r = self.nM_r
        index_m = self.nM_m +index_r
        index_t = self.nM_t + index_m
        index_c = self.nM_c + index_t
        mvec_tmp[        : index_r]=np.clip(
            mvec[        : index_r], np.log(self.resmin), np.log(self.resmax)
            )
        mvec_tmp[ index_r: index_m]=np.clip(
            mvec[ index_r: index_m], self.chgmin, self.chgmax
            )
        if self.nM_t > 0:
            mvec_tmp[ index_m: index_t]=np.clip(
                mvec[ index_m: index_t], np.log(self.taumin), np.log(self.taumax)
                )
        if self.nM_c > 0: 
            mvec_tmp[ index_t: index_c]=np.clip(
                mvec[ index_t: index_c], self.cmin, self.cmax
                )
        return mvec_tmp

    def Japprox(self, model_vector, perturbation=0.1, min_perturbation=1e-3):
        delta_m = min_perturbation  # np.max([perturbation*m.mean(), min_perturbation])
#        delta_m = perturbation  # np.max([perturbation*m.mean(), min_perturbation])
        J = []

        for i, entry in enumerate(model_vector):
            mpos = model_vector.copy()
            mpos[i] = entry + delta_m

            mneg = model_vector.copy()
            mneg[i] = entry - delta_m

            pos = self.predicted_data(mpos)
            neg = self.predicted_data(mneg)
            J.append((pos - neg) / (2. * delta_m))

        return np.vstack(J).T


    def get_Wd(self, dobs, ratio=0.10, plateau=0):
        std = np.sqrt(plateau**2 + (ratio*np.abs(dobs))**2) 
        Wd = np.diag(1 / std)
        self.Wd = Wd 
        return Wd

    def get_Ws(self):
        nM = self.nM
        Ws = np.eye(nM)
        self.Ws = Ws
        
        return Ws
    
    def compute_sensitivity(self,J):
        return  np.sqrt(np.sum(J**2, axis=0))

    def update_Ws(self, J):

        Wd=self.Wd
        threshold=self.Ws_threshold
        ind_r = self.nM_r
        ind_m = self.nM_m + ind_r
        ind_t = self.nM_t + ind_m
        ind_c = self.nM_c + ind_t

        Sensitivity = self.compute_sensitivity(Wd@J)
        # Sensitivity[:ind_r] /= Sensitivity[:ind_r].max()
        # Sensitivity[ind_r:ind_m] /= Sensitivity[ind_r:ind_m].max()
        # if self.nM_t > 0:
        #     Sensitivity[ind_m:ind_t] /= Sensitivity[ind_m:ind_t].max()
        # if self.nM_c > 0:
        #     Sensitivity[ind_t:ind_c] /= Sensitivity[ind_t:ind_c].max()
        # nlayer=self.nlayer
        #     Sensitivity[i*nlayer:(i+1)*nlayer] /= Sensitivity[i*nlayer:(i+1)*nlayer].max()
        Sensitivity /= Sensitivity.max()
        Sensitivity = np.clip(Sensitivity, threshold, 1)
#        Sensitivity += threshold * np.ones(nM)
        Ws = np.diag(Sensitivity)
        self.Ws = Ws
        return Ws

    def get_Wx(self):
        nlayer = self.nlayer
        depth = self.model_base["depth"]
        depth= np.r_[depth,2*depth[-1]-depth[-2]]
        if nlayer == 1:
            nM = self.nM
            nP = self.nP
            Wx = np.zeros((nM,nP))
            print("No smoothness for one layer model")
            self.Wx = Wx
            return Wx
        Wx_block = np.zeros((nlayer-1, nlayer))
        delta_x = np.diff(depth)
        elm1 = 1/delta_x[:-1]
        Wx_block[:,:-1] = -np.diag(elm1)
        Wx_block[:,1:] += np.diag(elm1)
        elm2 = np.sqrt(delta_x)
        Wx_block = Wx_block @ np.diag(elm2)
        # Wx_block = np.zeros((nlayer-1, nlayer))
        # Wx_block[:,1:-1] = np.eye(nlayer-1)
        # Wx_block[:,2:] -= np.eye(nlayer-1)
        Wx=np.block([
        [Wx_block, np.zeros((nlayer-1, nlayer*3))], # Resistivity
        [np.zeros((nlayer-1, nlayer*1)), Wx_block, np.zeros((nlayer-1, nlayer*2))], # Chargeability
        [np.zeros((nlayer-1, nlayer*2)), Wx_block, np.zeros((nlayer-1, nlayer*1))], # Time constant
        [np.zeros((nlayer-1, nlayer*3)), Wx_block], # Exponent C
        ])
        self.Wx = Wx
        return Wx

    def BetaEstimate_byEig(self,mvec, beta0_ratio, eig_tol=1e-12, update_Wsen=True):
        alphax=self.alphax
        alphas=self.alphas
        Wd = self.Wd
        Wx = self.Wx
        Ws= self.Ws
        mvec = self.clip_model(mvec)
        J = self.Japprox(mvec)

        if update_Wsen:
            Ws = self.update_Ws(J)
        # Prj_m = self.Prj_m  # Use `Proj_m` to map the model space

        # Effective data misfit term with projection matrix
        A_data =  J.T @ Wd.T @ Wd @ J 
        eig_data = np.linalg.eigvalsh(A_data)
        
        # Effective regularization term with projection matrix
        # A_reg = alphax* Prj_m.T @ Wx.T @ Wx @ Prj_m
        A_reg = alphax * Wx.T @ Wx 
        if Ws is not None:
            A_reg += alphas * (Ws.T @ Ws)
        eig_reg = np.linalg.eigvalsh(A_reg)
        
        # Ensure numerical stability (avoid dividing by zero)
        eig_data = eig_data[eig_data > eig_tol]
        eig_reg = eig_reg[eig_reg > eig_tol]

        # Use the ratio of eigenvalues to set beta range
        lambda_d = np.max(eig_data)
        lambda_r = np.min(eig_reg)
        return beta0_ratio * lambda_d / lambda_r

    def steepest_descent(self, dobs, model_init, niter):
        '''
        Eldad Haber, EOSC555, 2023, UBC-EOAS 
        '''
        model_vector = model_init
        r = dobs - self.predicted_data(model_vector)
        f = 0.5 * np.dot(r, r)

        error = np.zeros(niter + 1)
        error[0] = f
        model_itr = np.zeros((niter + 1, model_vector.shape[0]))
        model_itr[0, :] = model_vector

        print(f'Steepest Descent \n initial phid= {f:.3e} ')
        for i in range(niter):
            J = self.Japprox(model_vector)
            r = dobs - self.predicted_data(model_vector)
            dm = J.T @ r
            g = np.dot(J.T, r)
            Ag = J @ g
            alpha = np.mean(Ag * r) / np.mean(Ag * Ag)
            model_vector = self.constrain_model_vector(model_vector + alpha * dm)
            r = self.predicted_data(model_vector) - dobs
            f = 0.5 * np.dot(r, r)
            if np.linalg.norm(dm) < 1e-12:
                break
            error[i + 1] = f
            model_itr[i + 1, :] = model_vector
            print(f' i= {i:3d}, phid= {f:.3e} ')
        return model_vector, error, model_itr


    def Gradient_Descent(self, dobs, mvec_init, niter, beta, alphas, alphax,
            s0=1, sfac=0.5, stol=1e-6, gtol=1e-3, mu=1e-4, ELS=True, BLS=True ):
        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx
        mvec_old = mvec_init
        mvec_new = None
        mref = mvec_init
        error_prg = np.zeros(niter + 1)
        mvec_prg = np.zeros((niter + 1, mvec_init.shape[0]))
        rd = Wd @ (self.predicted_data(mvec_old) - dobs)
        phid = 0.5 * np.dot(rd, rd)
        rms = 0.5 * np.dot(Ws@(mvec_old - mref), Ws@(mvec_old - mref))
        rmx = 0.5 * np.dot(Wx @ mvec_old, Wx @ mvec_old)
        phim = alphas * rms + alphax * rmx
        f_old = phid + beta * phim
        k = 0
        error_prg[0] = f_old
        mvec_prg[0, :] = mvec_old
        print(f'Gradient Descent \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):
            # Calculate J:Jacobian and g:gradient
            J = self.Japprox(mvec_old)
            g = J.T @ Wd.T @ rd + beta * (alphas * Ws.T @ Ws @ (mvec_old - mref)
                                          + alphax * Wx.T @ Wx @ mvec_old)

            # Exact line search
            if ELS:
                t = np.dot(g,g)/np.dot(Wd@J@g,Wd@J@g)
#                t = (g.T@g)/(g.T@J.T@J@g)
            else:
                t = 1.

            # End inversion if gradient is smaller than tolerance
            g_norm = np.linalg.norm(g, ord=2)
            if g_norm < gtol:
                print(f"Inversion complete since norm of gradient is small as :{g_norm :.3e} ")
                break

            # Line search method Armijo using directional derivative
            s = s0
            dm = t*g
            directional_derivative = np.dot(g, -dm)

            mvec_new = self.proj_c(mvec_old - s * dm)
            rd = Wd @ (self.predicted_data(mvec_new) - dobs)
            phid = 0.5 * np.dot(rd, rd)
            rms = 0.5 * np.dot(Ws @ (mvec_new - mref), Ws @ (mvec_new - mref))
#            rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
            rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
            phim = alphas * rms + alphax * rmx
            f_new = phid + beta * phim
            if BLS:
                while f_new >= f_old + s * mu * directional_derivative:
                    s *= sfac
                    mvec_new = self.proj_c(mvec_old - s * dm)
                    rd = Wd @ (self.predicted_data(mvec_new) - dobs)
                    phid = 0.5 * np.dot(rd, rd)
                    rms = 0.5 * np.dot(Ws @ (mvec_new - mref), Ws @ (mvec_new - mref))
                    rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
                    phim = alphas * rms + alphax * rmx
                    f_new = phid + beta * phim
                    if np.linalg.norm(s) < stol:
                        break
            mvec_old = mvec_new
            mvec_prg[i + 1, :] = mvec_new
            f_old = f_new
            error_prg[i + 1] = f_new
            k = i + 1
            print(f'{k:3}, s:{s:.2e}, gradient:{g_norm:.2e}, phid:{phid:.2e}, phim:{phim:.2e}, f:{f_new:.2e} ')
        # filter model prog data
        mvec_prg = mvec_prg[:k]
        error_prg = error_prg[:k]
        # Save Jacobian
        self.Jacobian = J
        return mvec_new, error_prg, mvec_prg

    def GaussNewton_smooth(self, dobs, mvec_init, niter,
         beta0, coolingFactor=1.0, coolingRate=3, update_Wsen=True,
         s0=1, sfac=0.5, stol=1e-6, gtol=1e-3, mu=1e-4):
        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx
        alphas = self.alphas
        alphax = self.alphax
        # Prj_m = self.Prj_m
        # m_fix = self.m_fix
        mvec_old = mvec_init
        # applay initial mvec for reference mode
        m_ref = self.m_ref
        # get noise part
        # Initialize object function
        rd = Wd @ (self.predicted_data(mvec_old) - dobs)
        phid = 0.5 * np.dot(rd, rd)
        rms = 0.5 * np.linalg.norm(Ws @ (mvec_old - m_ref))**2
        rmx = 0.5 * np.linalg.norm(Wx @ mvec_old)**2
        # rmx = 0.5 * np.linalg.norm(Wx @ (m_fix + Prj_m@ mvec_old))**2
        phim = alphas * rms + alphax * rmx
        f_old = phid + beta0 * phim
        # Prepare array for storing error and model in progress
        error_prg = np.zeros(niter + 1)
        betas =  np.r_[beta0, np.zeros(niter)]
        mvec_prg = np.zeros((niter + 1, mvec_init.shape[0]))
        error_prg[0] = f_old
        mvec_prg[0, :] = mvec_old

        print(f'Gauss-Newton \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):
            beta = beta0 / (coolingFactor ** (i // coolingRate))
            mvec_old = self.clip_model(mvec_old)
            # Jacobian
            J = self.Japprox(mvec_old)

            if update_Wsen:
                Ws = self.update_Ws(J)

            # gradient
            # g = J.T @ Wd.T @ rd + beta * (alphas * Ws.T @ Ws @ (mvec_old - mref)
            #                               + alphax * Wx.T @ Wx @ mvec_old)
            g = J.T @ Wd.T @ rd
            g += beta * alphas * (Ws.T@Ws@ (mvec_old - m_ref))
            # g += beta * alphax * (m_fix.T@Wx.T @ Wx @ Prj_m +
            #                      Prj_m.T @Wx.T @ Wx @ Prj_m@ mvec_old)
            g += beta * alphax * (Wx.T @ Wx @ mvec_old)
            # g = g_d + beta *g_m

           # Hessian approximation
            H = J.T @ Wd.T @ Wd @ J  #+ beta * (alphas * Ws.T @ Ws + alphax * Wx.T @ Wx)
            # H_d = J.T @ Wd.T @ Wd @ J 
            H += beta * alphas * Ws.T @ Ws
            # H += beta * alphax * Prj_m.T@Wx.T @ Wx@Prj_m
            H += beta *  alphax *  Wx.T @ Wx
            # H = H_d + beta *H_m

            # model step
            dm = np.linalg.solve(H, g)

            # End inversion if gradient is smaller than tolerance
            g_norm = np.linalg.norm(g, ord=2)
            if g_norm < gtol:
                print(f"Inversion complete since norm of gradient is small as :{g_norm :.3e} ")
                break

            # update object function
            s = s0
            mvec_new = self.clip_model(mvec_old - s * dm)
            rd = Wd @ (self.predicted_data(mvec_new) - dobs)
            phid = 0.5 * np.dot(rd, rd)
            rms = 0.5 * np.linalg.norm(Ws @ (mvec_new - m_ref))**2
            # rmx = 0.5 * np.linalg.norm(Wx @ (m_fix+ Prj_m@mvec_new))**2
            rmx = 0.5 * np.linalg.norm(Wx @ mvec_new)**2
            phim = alphas * rms + alphax * rmx
            f_new = phid + beta * phim

            # Backtracking method using directional derivative Amijo
            directional_derivative = np.dot(g, -dm)
            while f_new >= f_old + s * mu * directional_derivative:
                # backtracking
                s *= sfac
                # update object function
                mvec_new = self.clip_model(mvec_old - s * dm)
                rd = Wd @ (self.predicted_data(mvec_new) - dobs)
                phid = 0.5 * np.dot(rd, rd)
                rms = 0.5 * np.linalg.norm(Ws @ (mvec_new - m_ref))**2
                rmx = 0.5 * np.linalg.norm(Wx @  mvec_new)**2
                # rmx = 0.5 * np.linalg.norm(Wx @ (m_fix+ Prj_m@mvec_new))**2
                phim = alphas * rms + alphax * rmx
                f_new = phid + beta * phim
                # Stopping criteria for backtrackinng
                if s < stol:
                    break

            # Update model
            mvec_old = mvec_new
            betas[i+1] = beta
            mvec_prg[i + 1, :] = mvec_new
            f_old = f_new
            error_prg[i + 1] = f_new
            k = i + 1
            print(f'{k:3}, beta:{beta:.1e}, step:{s:.1e}, gradient:{g_norm:.1e}, phid:{phid:.1e}, phim:{phim:.1e}, f:{f_new:.1e} ')
        # clip progress of model and error in inversion
        error_prg = error_prg[:k+1]
        mvec_prg = mvec_prg[:k+1]
        betas = betas[:k+1]
        return mvec_new, mvec_prg, betas

    def objec_func(self,mvec,dobs,beta):
        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx
        alphas = self.alphas
        alphax = self.alphax
        m_ref = self.m_ref
        Prj_m = self.Prj_m
        m_fix = self.m_fix
        rd = Wd @ (self.predicted_data(mvec) - dobs)
        phid = 0.5 * np.dot(rd, rd)
        rms =  0.5 * np.linalg.norm(Ws @ (mvec - m_ref))**2
        rmx = 0.5 * np.dot(Wx @ mvec, Wx @ mvec)
        # rmx = 0.5 * np.linalg.norm(Wx @ (m_fix+ Prj_m@mvec))**2
        phim = alphas * rms + alphax * rmx
        f_obj = phid + beta * phim
        return f_obj, phid, phim

    def plot_model(self, model, depth_min=-1e3,depth_max=1e3, ax=None, **kwargs):
        """
        Plot a single model (e.g., resistivity, chargeability) with depth.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # Default plotting parameters
        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "label": "model",
        }
        default_kwargs.update(kwargs)

        # Prepare depth and model data for plotting
        depth = np.r_[depth_min + self.model_base["depth"][0], 
                      self.model_base["depth"],
                      depth_max + self.model_base["depth"][-1] ]
        depth_plot = np.vstack([depth, depth]).flatten(order="F")[1:-1]
 #       depth_plot = np.hstack([depth_plot, depth_plot[-1] * 1.5])  # Extend depth for plot
        model_plot = np.vstack([model, model]).flatten(order="F")

        # Plot model with depth
        ax.plot(model_plot, depth_plot, **default_kwargs)
        return ax
    
    def plot_IP_par(self, mvec, ax=None, label=None, **kwargs):
        """
        Plot all IP parameters (resistivity, chargeability, time constant, exponent c).
        """
        if ax is None:
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))  # Create 2x2 grid of subplots
        else:
            ax = np.array(ax)  # Convert ax to a NumPy array if it's not already
            ax = ax.flatten()  # Ensure ax is a flat array


        # Convert model vector to parameters
        model = self.get_ip_model(mvec)

        # Plot each model parameter
        self.plot_model(model["res"], ax=ax[0], label=label, **kwargs)
        ax[0].set_title("Resistivity (ohm-m)")

        self.plot_model(model["m"], ax=ax[1], label=label, **kwargs)
        ax[1].set_title("Chargeability")

        self.plot_model(model["tau"], ax=ax[2], label=label, **kwargs)
        ax[2].set_title("Time Constant (sec)")

        self.plot_model(model["c"], ax=ax[3], label=label, **kwargs)
        ax[3].set_title("Exponent C")
        return ax

class InducedPolarization:

    def __init__(self,
        res0=None, con8=None, eta=None, tau=None, c=None,
        freq=None, times=None, windows_strt=None, windows_end=None
        ):

        if res0 is not None and con8 is not None and eta is not None:
            assert np.allclose(con8 * res0 * (1 - eta), 1.)
        self.con8 = con8
        self.res0 = res0
        self.eta = eta
        if self.res0 is None and self.con8 is not None and self.eta is not None:
            self.res0 = 1./ (self.con8 * (1. - self.eta))
        if self.res0 is not None and self.con8 is None and self.eta is not None:
            self.con8 = 1./ (self.res0 * (1. - self.eta))
        self.tau = tau
        self.c = c
        self.freq = freq
        self.times = times
        self.windows_strt = windows_strt
        self.windows_end = windows_end

    def validate_times(self, times):
        assert np.all(times >= -eps ), "All time values must be non-negative."
        if len(times) > 1:
            assert np.all(np.diff(times) >= 0), "Time values must be in ascending order."
    
    def get_param(self, param, default):
        return param if param is not None else default

    def pelton_res_f(self, freq=None, res0=None, eta=None, tau=None, c=None):
        freq = self.get_param(freq, self.freq)
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        c = self.get_param(c, self.c)
        iwtc = (1.j * 2. * np.pi * freq*tau) ** c
        return res0*(1.-eta*(1.-1./(1. + iwtc)))

    def pelton_con_f(self, freq=None, con8=None, eta=None, tau=None, c=None):
        freq = self.get_param(freq, self.freq)
        con8 = self.get_param(con8, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        c = self.get_param(c, self.c)
        iwtc = (1.j * 2. * np.pi * freq*tau) ** c
        return con8-con8*(eta/(1.+(1.-eta)*iwtc))

    def debye_con_t(self, times=None, con8=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        con8 = self.get_param(con8, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        debye = np.zeros_like(times)
        ind_0 = (times == 0)
        debye[ind_0] = 1.0
        debye[~ind_0] = -eta/((1.0-eta)*tau)*np.exp(-times[~ind_0]/((1.0-eta)*tau))
        return con8*debye

    def debye_con_t_intg(self, times=None, con8=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        con8 = self.get_param(con8, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        return con8 *(1.0 -eta*(1. -np.exp(-times/((1.0-eta)*tau))))

    def debye_res_t(self, times=None, res0=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        debye = np.zeros_like(times)
        res8 = res0 * (1.0 - eta)
        ind_0 = (times == 0)
        debye[ind_0] = res8 
        debye[~ind_0] = (res0-res8)/tau * np.exp(-times[~ind_0] / tau)
        return debye

    def debye_res_t_intg(self, times=None, res0=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        res8 = res0 * (1.0 - eta)
        return res8 + (res8 - res0)*(np.exp(-times/tau) - 1.0)

    def freq_symmetric(self,f):
        symmetric = np.zeros_like(f, dtype=complex)
        nstep = len(f)
        half_step = nstep // 2
        symmetric[:half_step] = f[:half_step]
        symmetric[half_step:] = f[:half_step].conj()[::-1]
        assert np.allclose(symmetric[:half_step].real, symmetric[half_step:].real[::-1])
        assert np.allclose(symmetric[:half_step].imag, -symmetric[half_step:].imag[::-1])
        return symmetric

    def get_frequency_tau(self, tau=None, log2nfreq=16): 
        tau = self.get_param(tau, self.tau)
        log2nfreq = int(log2nfreq)
        nfreq = 2**log2nfreq
        freqcen = 1 / tau
        freqend = freqcen * nfreq**0.5
        freqstep = freqend / nfreq
        freq = np.arange(0, freqend, freqstep)
        self.freq = freq
        print(f'log2(len(freq)) {np.log2(len(freq))} considering tau')
        return freq

    def get_frequency_tau2(self, tau=None, log2min=-8, log2max=8):
        tau = self.get_param(tau, self.tau)
        freqcen = 1 / tau
        freqend = freqcen * 2**log2max
        freqstep = freqcen * 2**log2min
        freq = np.arange(0, freqend, freqstep)
        self.freq = freq
        print(f'log2(len(freq)) {np.log2(len(freq))} considering tau')
        return freq


    def get_frequency_tau_times(self, tau=None, times=None,log2min=-8, log2max=8):
        tau = self.get_param(tau, self.tau)
        times = self.get_param(times, self.times)
        self.validate_times(times)
        _, windows_end = self.get_windows(times)

        freqstep = 1/tau*(2**np.floor(np.min(
            np.r_[log2min,np.log2(tau/windows_end[-1])]
        )))
        freqend = 1/tau*(2**np.ceil(np.max(
            np.r_[log2max, np.log2(2*tau/min(np.diff(times)))]
        )))
        freq = np.arange(0,freqend,freqstep)
        self.freq=freq
        print(f'log2(freq) {np.log2(len(freq))} considering tau and times')
        return freq

    def compute_fft(self, fft_f, freqend, freqstep):
        fft_f = self.freq_symmetric(fft_f)
        fft_data = fftpack.ifft(fft_f).real * freqend
        fft_times = np.fft.fftfreq(len(fft_data), d=freqstep)
        return fft_times[fft_times > -eps], fft_data[fft_times > -eps]

    def pelton_fft(self, con_form=True, con8=None, res0=None, eta=None, tau=None, c=None, freq=None):
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        c = self.get_param(c, self.c) 
        freq = self.get_param(freq, self.freq) 
        freqstep = freq[1] - freq[0]
        freqend = freq[-1] +freqstep

        if con_form:
            con8 = self.get_param(con8, self.con8)
            fft_f = self.pelton_con_f(freq=freq,
                     con8=con8, eta=eta, tau=tau, c=c)
        else:
            res0 = self.get_param(res0, self.res0)
            fft_f = self.pelton_res_f(freq=freq,
                     res0=res0, eta=eta, tau=tau, c=c)
        fft_times, fft_data = self.compute_fft(fft_f, freqend, freqstep)
        return fft_times, fft_data

    def get_windows(self, times):
        self.validate_times(times)
        windows_strt = np.zeros_like(times)
        windows_end = np.zeros_like(times)
        dt = np.diff(times)
        windows_strt[1:] = times[:-1] + dt / 2
        windows_end[:-1] = times[1:] - dt / 2
        windows_strt[0] = times[0] - dt[0] / 2
        windows_end[-1] = times[-1] + dt[-1] / 2
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        return windows_strt,windows_end

    def apply_windows(self, times, data, windows_strt=None, windows_end=None):
        if windows_strt is None:
            windows_strt = self.windows_strt
        if windows_end is None:
            windows_end = self.windows_end
        self.validate_times(times)

        # Find bin indices for start and end of each window
        start_indices = np.searchsorted(times, windows_strt, side='left')
        end_indices = np.searchsorted(times, windows_end, side='right')

        # Compute windowed averages
        window_data = np.zeros_like(windows_strt, dtype=float)
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            if start < end:  # Ensure there are elements in the window
                window_data[i] = np.mean(data[start:end])

        return window_data

    def get_window_matrix (self, times, windows_strt=None, windows_end=None):
        windows_strt = self.get_param(windows_strt, self.windows_strt)
        windows_end = self.get_param(windows_end, self.windows_end)
        self.validate_times(times)
        nwindows = len(windows_strt)
        window_matrix = np.zeros((nwindows, len(times)))
        for i in range(nwindows):
            start = windows_strt[i]
            end = windows_end[i]
            ind_time = (times >= start) & (times <= end)
            if ind_time.sum() > 0:
                window_matrix[i, ind_time] = 1/ind_time.sum()
        return window_matrix    
    

class TEM_Signal_Process:
    
    def __init__(self,  
        base_freq,on_time, rmp_time, rec_time, smp_freq,
        windows_cen=None, windows_strt = None, windows_end = None):
        self.base_freq = base_freq
        self.on_time = on_time
        self.rmp_time = rmp_time
        self.rec_time = rec_time
        self.smp_freq = smp_freq
        time_step = 1./smp_freq
        self.time_step = time_step
        self.times_rec = np.arange(0,rec_time,time_step) + time_step
        self.times_filt = np.arange(0,rec_time,time_step)
        self.windows_cen= windows_cen
        self.windows_strt = windows_strt
        self.windows_end = windows_end
    
    def get_param(self, param, default):
        return param if param is not None else default

    def validate_times(self, times):
        if len(times) > 1:
            assert np.all(np.diff(times) >= 0), "Time values must be in ascending order."
    
    def get_param(self, param, default):
        return param if param is not None else default

    def get_windows_cen(self, windows_cen):
        self.validate_times( windows_cen)
        self.windows_cen = windows_cen
        windows_strt = np.zeros_like( windows_cen)
        windows_end = np.zeros_like( windows_cen)
        dt = np.diff( windows_cen) + 2*eps
        windows_strt[1:] =  windows_cen[:-1] + dt / 2
        windows_end[:-1] =  windows_cen[1:] - dt / 2
        windows_strt[0] =  windows_cen[0] - dt[0] / 2
        windows_end[-1] =  windows_cen[-1] + dt[-1] / 2
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        return windows_strt,windows_end

    def get_window_linlog(self,linstep,time_trns):
        rmp_time = self.rmp_time
        rec_time = self.rec_time + rmp_time
        nlinstep = round(time_trns/linstep)
        logstep = np.log((linstep+time_trns)/time_trns)
        logstrt = np.log(time_trns)
#        logend = np.log(rec_time) + logstep + eps
        logend = np.log(rec_time) - eps
        nlogstep = round((logend-logstrt)/logstep)
        windows_cen= np.r_[np.arange(0,time_trns,linstep), np.exp(np.arange(logstrt,logend,logstep))]
        windows_strt = np.r_[np.arange(0,time_trns,linstep)-linstep/2, np.exp(np.arange(logstrt-logstep/2,logend-logstep/2,logstep))]
        windows_end =  np.r_[np.arange(0,time_trns,linstep)+linstep/2,  np.exp(np.arange(logstrt+logstep/2,logend+logstep/2,logstep))]
        self.windows_cen = windows_cen
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        print(f'linear step: {nlinstep}, log step: {nlogstep}, total steps: {nlinstep+nlogstep}')
        return windows_cen, windows_strt, windows_end

    def get_window_log(self,logstep, tstart, tend=None, rmp_time=None):
        tend = self.get_param(tend, self.rec_time)
        rmp_time = self.get_param(rmp_time, self.rmp_time)
        if rmp_time is not None:
            tstart = tstart
            tend = tend - rmp_time     

        logstrt = np.log10(tstart)
        logend = np.log10(tend)
        log10_windows_cen = np.arange(logstrt,logend,logstep)
#        log10_windows_cen = np.linspace(logstrt,logend,logstep)
        windows_cen  = 10.**log10_windows_cen +rmp_time
        windows_strt = 10.**(log10_windows_cen-logstep/2) +rmp_time
        windows_end  = 10.**(log10_windows_cen+logstep/2) +rmp_time
        # windows_strt = 10.**(np.arange(logstrt-logstep/2,logend-logstep/2,logstep))
        # windows_end  = 10.**(np.arange(logstrt+logstep/2,logend+logstep/2,logstep))
        # print(f'log step: {len(self.windows_cen_log)}')
        self.windows_cen = windows_cen 
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        return windows_cen, windows_strt, windows_end

    def window(self,times,data, windows_strt=None, windows_end=None):
        windows_strt = self.get_param(windows_strt, self.windows_strt)
        windows_end = self.get_param(windows_end, self.windows_end)
        self.validate_times(times)

        # Find bin indices for start and end of each windows
        start_indices = np.searchsorted(times, windows_strt, side='left')
        end_indices = np.searchsorted(times, windows_end, side='right')

        # Compute windowed averages
        data_window = np.zeros_like(windows_strt, dtype=float)
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            if start < end:  # Ensure there are elements in the window
                data_window[i] = np.mean(data[start:end])
        return data_window
    
    def get_window_matrix (self, times, windows_strt=None, windows_end=None):
        windows_strt = self.get_param(windows_strt, self.windows_strt)
        windows_end = self.get_param(windows_end, self.windows_end)
        self.validate_times(times)
        nwindows = len(windows_strt)
        window_matrix = np.zeros((nwindows, len(times)))
        for i in range(nwindows):
            start = windows_strt[i]
            end = windows_end[i]
            ind_time = (times >= start) & (times <= end)
            if ind_time.sum() > 0:
                window_matrix[i, ind_time] = 1/ind_time.sum()
        return window_matrix

    def plot_window_data(self,data=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        windows_strt= self.windows_strt
        windows_end = self.windows_end
        windows_cen = self.windows_cen
        if data is None:
            ax.loglog(windows_cen, windows_cen,"k*")
            ax.loglog(windows_strt, windows_cen,"b|")
            ax.loglog(windows_end, windows_cen,"m|")
        else:
            assert len(data) == len(self.windows_cen), "Data and windows must have the same length."
            ax.loglog(windows_cen, data,"k*")
            ax.loglog(windows_strt, data,"b|")
            ax.loglog(windows_end, data,"m|")
        ax.grid(True, which="both")
        ax.legend(["center","start","end"])
        return ax

    def butter_lowpass(self, cutoff, order=1):
        fs = self.smp_freq
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
   
    def apply_lowpass_filter(self, data, cutoff, order=1):
        b, a = self.butter_lowpass(cutoff, order=order)
        y = filtfilt(b, a, data)
        return y
    
    def filter_linear_rmp(self, rmp_time=None, times_rec=None, time_step=None):
        rmp_time  = self.get_param(rmp_time, self.rmp_time)
        times_rec = self.get_param(times_rec, self.times_rec)
        time_step = self.get_param(time_step, self.time_step)
        times_filt = self.times_filt
        filter_linrmp = np.zeros_like(times_filt)
        inds_rmp = times_filt <= rmp_time
        filter_linrmp[inds_rmp] =   1.0/float(inds_rmp.sum())
        return filter_linrmp

    def filter_linear_rmp_rect(self, rmp_time=None):
        if rmp_time is None:
            rmp_time = self.rmp_time
        pos_off = self.filter_linear_rmp(rmp_time=rmp_time)
        return np.r_[-pos_off, pos_off]
        
    def rect_wave(self, t, base_freq=None, neg=False):
        self.get_param(base_freq, self.base_freq)
        if neg:
            pos= 0.5*(1.0+signal.square(2*np.pi*(base_freq*t    ),duty=0.25))
            neg=-0.5*(1.0+signal.square(2*np.pi*(base_freq*t+0.5),duty=0.25))
            return pos + neg
        else :
            pos= 0.5*(1.0+signal.square(2*np.pi*(base_freq*t    ),duty=0.5))
            return pos

    def rect_wave_rmp(self, t, base_freq=None, rmp_time=None,neg=False):
        self.get_param(base_freq, self.base_freq)
        self.get_param(rmp_time, self.rmp_time)
        if neg:
            print("under construction")
            return None
        else :
            pos= 0.5*(1.0+signal.square(2*np.pi*(base_freq*t    ),duty=0.5))
            ind_pos_on = t<=rmp_time
            pos[ind_pos_on] = t/rmp_time
            ind_pos_off = (t>=0.5/base_freq) & (t<=0.5/base_freq+rmp_time)
            pos[ind_pos_off] = 1.0 - (t-0.5/base_freq)/rmp_time
            return pos


    def interpolate_data_lin(self,times,data, times_rec=None,method='cubic'):
        '''
        times (array-like): Original time points (not uniformly spaced).
        data (array-like): Original data values at time points `t`.
        method (str): Interpolation method ('linear', 'nearest', 'cubic', etc.).
        Returns:
            resampled_data (np.ndarray): Resampled data on `t_new`.
        '''
        times_rec = self.get_param(times_rec, self.times_rec)
        interpolator = interp1d(
            x=times,
            y=data,
            kind=method,
            fill_value='extrapolate'
        )
        return interpolator(times_rec)


    def interpolate_data(self,times,data, times_rec=None,method='cubic',
        logmin_time=1e-8, linScale_time=1.0, logmin_data=1e-8, linScale_data=1.0):
        '''
        times (array-like): Original time points (not uniformly spaced).
        data (array-like): Original data values at time points `t`.
        method (str): Interpolation method ('linear', 'nearest', 'cubic', etc.).
        Returns:
            resampled_data (np.ndarray): Resampled data on `t_new`.
        '''
        times_rec = self.get_param(times_rec, self.times_rec)
        pslog_time = PsuedoLog(logmin=logmin_time, linScale=linScale_time)
        pslog_data = PsuedoLog(logmin=logmin_data, linScale=linScale_data)
        if method == "linear":
            interpolator = interp1d(
                x=pslog_time.pl_value(times),
                y=pslog_data.pl_value(data),
                kind=method,
                fill_value='extrapolate'
            )
        if method == "cubic":
            interpolator = CubicSpline(
                x=pslog_time.pl_value(times),
                y=pslog_data.pl_value(data),
            )
        
        return pslog_data.pl_to_linear(interpolator(pslog_time.pl_value(times_rec)))

    def deconvolve(self, data, data_pulse):
        filt, reminder = signal.deconvolve(
            np.r_[data, np.zeros(len(data)-1)],
            data_pulse
            )
        print(reminder)
        print(np.linalg.norm(reminder))
        return filt

class PsuedoLog:
    def __init__(self, logmin=None, linScale=None, max_y=eps, min_y=-eps,
        logminx=None, linScalex=None, max_x=eps, min_x=-eps):
        self.logmin = logmin
        self.linScale = linScale
        self.max_y = max_y
        self.min_y = min_y
        self.logminx = logminx
        self.linScalex = linScalex
        self.max_x = max_x
        self.min_x =min_x

    def get_param(self, param, default):
        return param if param is not None else default

    def pl_value(self, lin, logmin=None, linScale=None):    
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        # Check if `lin` is scalar
        is_scalar = np.isscalar(lin)
        if is_scalar:
            lin = np.array([lin])  # Convert scalar to array for uniform processing
                
        abs_lin = np.abs(lin)
        sign_lin = np.sign(lin)
        ind_pl = (abs_lin >= logmin)
        ind_lin = ~ind_pl
        plog = np.zeros_like(lin)
        plog[ind_pl] = sign_lin[ind_pl] * (
            np.log10(abs_lin[ind_pl] / logmin) + linScale
            )
        plog[ind_lin] = lin[ind_lin] / logmin * linScale
        return plog
    
    def pl_to_linear(self,plog, logmin=None, linScale=None):   
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        # Check if `lin` is scalar
        is_scalar = np.isscalar(plog)
        if is_scalar:
            lin = np.array([plog])  # Convert scalar to array for uniform processing
        abs_plog = np.abs(plog)
        sign_plog = np.sign(plog)
        ind_pl = (abs_plog >= linScale)
        ind_lin = ~ind_pl
        lin = np.zeros_like(plog)
        lin[ind_pl] = sign_plog[ind_pl] * logmin * 10 ** (abs_plog[ind_pl] - linScale)
        lin[ind_lin] = plog[ind_lin] / linScale * logmin
        return lin

    def semiply(self, x, y, logmin=None, linScale=None, ax=None, xscale_log=True,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        logmin = self.get_param(logmin, self.logmin)
        linScale = self.get_param(linScale, self.linScale)
        plog_y = self.pl_value(lin=y, logmin=logmin, linScale=linScale)
        
        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "markersize": 1,
        }
        default_kwargs.update(kwargs)
        if xscale_log:
            ax.semilogx(x, plog_y, **default_kwargs)
        else:
            ax.plot(x, plog_y, **default_kwargs)
        
        self.max_y = np.max(np.r_[self.max_y,np.max(y)])
        self.min_y = np.min(np.r_[self.min_y,np.min(y)])
        return ax


    def semiplx(self, x, y,logminx=None,linScalex=None,ax=None, yscale_log=True,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        logminx = self.get_param(logminx, self.logminx)    
        linScalex = self.get_param(linScalex, self.linScalex)
        plog_x = self.pl_value(lin=x, logmin=logminx, linScale=linScalex)

        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "markersize": 1,
        }
        default_kwargs.update(kwargs)
        if yscale_log:
            ax.semilogy(plog_x, y, **default_kwargs)
        else:
            ax.plot(plog_x, y, **default_kwargs)
        self.max_x = np.max(np.r_[self.max_x,np.max(x)])
        self.min_x = np.min(np.r_[self.min_x,np.min(x)])
        return ax

    def plpl_plot(self, x, y,
        logminx=None,linScalex=None,logmin=None,linScale=None,ax=None,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        logminx = self.get_param(logminx, self.logminx)
        linScalex = self.get_param(linScalex, self.linScalex)
        plog_x = self.pl_value(lin=x, logmin=logminx, linScale=linScalex)
        plog_y = self.pl_value(lin=y, logmin=logmin, linScale=linScale)

        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "markersize": 1,
        }
        default_kwargs.update(kwargs)
        ax.plot(plog_x, plog_y, **default_kwargs)
        self.max_y = np.max(np.r_[self.max_y,np.max(y)])
        self.min_y = np.min(np.r_[self.min_y,np.min(y)])
        self.max_x = np.max(np.r_[self.max_x,np.max(x)])
        self.min_x = np.min(np.r_[self.min_x,np.min(x)])
        return ax

    def pl_axes(self,ax,logmin=None,linScale=None,max_y=None,min_y=None):
        assert hasattr(ax, 'set_xlim') and hasattr(ax, 'set_xticks') and hasattr(ax, 'set_xticklabels'), \
        "Provided 'ax' is not a valid Matplotlib Axes object."
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        max_y = self.get_param(max_y, self.max_y)
        min_y= self.get_param(min_y, self.min_y)

        if max_y <= logmin:
            n_postick = 1
        else:
            n_postick= int(np.ceil(np.log10((max_y+eps)/logmin)+1))
        posticks = linScale + np.arange(n_postick)
        #poslabels = logmin*10**np.arange(n_postick)
        poslabels = [f"{v:.0e}" for v in (logmin * 10**np.arange(n_postick))]

        if -min_y <= logmin:
            n_negtick = 1
        else:
            n_negtick = int(np.ceil(np.log10((-min_y+eps)/logmin)+1))

        negticks = -linScale - np.arange(n_negtick)
        negticks = negticks[::-1]
        #neglabels = -logmin*10**np.arange(n_negtick)
        neglabels = [f"{v:.0e}" for v in (-logmin * 10**np.arange(n_negtick))[::-1]]
#        neglabels = neglabels[::-1]
#        ticks  = np.hstack(( negticks, [0], posticks))
        ticks  = np.r_[negticks, 0, posticks]
        labels = np.hstack((neglabels, [0], poslabels))
        ax.set_ylim([min(ticks), max(ticks)])
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        # reset max and min
        self.max_y = eps
        self.min_y = -eps
        return ax

    def pl_axes_x(self,ax,logminx=None,linScalex=None,max_x=None,min_x=None):
        assert hasattr(ax, 'set_xlim') and hasattr(ax, 'set_xticks') and hasattr(ax, 'set_xticklabels'), \
        "Provided 'ax' is not a valid Matplotlib Axes object."
        logminx = self.get_param(logminx, self.logminx)    
        linScalex = self.get_param(linScalex, self.linScalex)
        max_x = self.get_param(max_x, self.max_x)
        min_x= self.get_param(min_x, self.min_x)
        if max_x <= logminx:
            n_postick = 1
        else:
            n_postick= int(np.ceil(np.log10(max_x/logminx)+1))
        posticks = linScalex + np.arange(n_postick)
        poslabels = [f"{v:.0e}" for v in (logminx * 10**np.arange(n_postick))]
        if -min_x <= logminx:
            n_negtick = 1
        else:
            n_negtick = int(np.ceil(np.log10(-min_x/logminx)+1))
        negticks = -linScalex - np.arange(n_negtick)
        negticks = negticks[::-1]
        neglabels = [f"{v:.0e}" for v in (-logminx * 10**np.arange(n_negtick))[::-1]]
        ticks  = np.r_[negticks, 0, posticks]
        labels = np.hstack((neglabels, [0], poslabels))
        ax.set_xlim([min(ticks), max(ticks)])
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        # reset max and min
        self.max_x = eps
        self.min_x = -eps
        return ax
    
    def pl_axvline(self, ax, x, **kwargs):
        logminx = self.logminx
        linScalex = self.linScalex
        default_kwargs = {
            "linestyle": "--",
            "color": "gray",
            "linewidth": 1.0,
        }
        default_kwargs.update(kwargs)
        ax.axvline(self.pl_value(x,logmin=logminx, linScale=linScalex), **default_kwargs)
        return ax
    
    def pl_axhline(self, ax, y, **kwargs):
        logmin = self.logmin
        linScale = self.linScale
        default_kwargs = {
            "linestyle": "--",
            "color": "gray",
            "linewidth": 1.0,
        }
        default_kwargs.update(kwargs)
        ax.axhline(self.pl_value(y, logmin=logmin,linScale=linScale), **default_kwargs)
        return ax

def solve_polynomial(a, n,pmax):
    # Coefficients of the polynomial -x^{n+1} + (1+a)x - a = 0
    coeffs = [-1] + [0] * (n-1) + [(1 + a), -a]  # [-1, 0, ..., 0, (1 + a), -a]
    
    # Find the roots of the polynomial
    roots = np.roots(coeffs)
    
    # Filter real roots
    real_roots = [r.real for r in roots if np.isreal(r)]
    
    # Find the real root closest to pmax
    if real_roots:
        closest_root = real_roots[np.argmin(np.abs(np.array(real_roots) - pmax))]
        return closest_root
    else:
        return None  # Return None if no real roots are found

def mesh_Pressure_Vessel(tx_radius,cs1,ncs1, pad1max,cs2,max,lim,pad2max): 
    h1a = discretize.utils.unpack_widths([(cs1, ncs1)])
    a1 = (tx_radius- np.sum(h1a))/cs1 
    n_tmp = -1 + np.log((a1+1)*pad1max-a1)/np.log(pad1max)
    npad1b= int(np.ceil(n_tmp))
    pad1 = solve_polynomial(a1, npad1b, pad1max)
    npad1c = int(np.floor(np.log(cs2/cs1)/np.log(pad1))-npad1b)
    if npad1c< 0:
        print("error: padx1max is too large")

    h1bc = discretize.utils.unpack_widths([(cs1, npad1b+npad1c, pad1)])

    ncs2 = int(np.ceil( (max-np.sum(np.r_[h1a,h1bc])) / cs2 ))

    h2a= discretize.utils.unpack_widths([(cs2, ncs2)])

    a2 = (lim-np.sum(np.r_[h1a, h1bc, h2a]))/cs2 
    n_tmp = -1 + np.log((a2+1)*pad2max-a2)/np.log(pad2max)
    npad2 = int(np.ceil(n_tmp))
    pad2 = solve_polynomial(a2, npad2, pad2max)
    h2b = discretize.utils.unpack_widths([(cs2, npad2, pad2)])
    h = np.r_[h1a,h1bc,h2a,h2b]
    return h
