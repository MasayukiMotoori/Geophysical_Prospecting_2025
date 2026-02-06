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
from IPython.display import display, Latex, Markdown


import  os
eps= np.finfo(float).eps
import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from abc import ABC, abstractmethod

class TorchHelper:
    @staticmethod
    def to_tensor_r(x, dtype=torch.float32, device='cpu'):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype, device=device)
        else:
            return torch.tensor(x, dtype=dtype, device=device)

    @staticmethod
    def to_tensor_c(x, dtype=torch.complex64, device='cpu'):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype, device=device)
        else:
            return torch.tensor(x, dtype=dtype, device=device)
    @staticmethod
    def to_numpy_r(x):
        """
        Convert to numpy and FORCE real values.
        Imaginary part is explicitly discarded.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)

        # Explicitly handle complex numbers
        if np.iscomplexobj(x):
            return np.real(x)
        else:
            return x.astype(float)

    @staticmethod
    def to_numpy_c(x):
        """
        Convert to numpy, preserving complex values if present.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            return np.asarray(x)


class Pelton_res_f():
    def __init__(self, freq=None, con=False,
            reslim= [1e-2,1e5],
            chglim= [1e-3, 0.9],
            taulim= [1e-8, 1e4],
            clim= [0.2, 0.8]
                ):
        self.freq = TorchHelper.to_tensor_c(freq) if freq is not None else None
        self.reslim = TorchHelper.to_tensor_r(np.log(reslim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.clim = TorchHelper.to_tensor_r(clim)
        self.con = con

    def show_equation(self):
        display(Markdown(r"""
### Pelton Resistivity Model

$$
\rho(\omega) = \rho_0 \left[ 1 - \eta \left(1 - \frac{1}{1+(i\omega\tau)^c} \right) \right] = 
\rho_0 \left[ \frac{\tau^{-c} + (1-\eta)(i\omega)^c}{\tau^{-c} + (i\omega)^c} \right]
$$

- $\rho_0$: Resistivity at low frequency ($\Omega\,$m)  
- $\eta$: Chargeability (dimensionless)  
- $\tau$: Time constant (s)  
- $c$: Exponent C
"""))

    def f(self,p):
        """
        - p` (`torch.Tensor`): Model parameters in real number
        - p[0]: log of Resistivity at low frequency
        - p[1]: Chargeability  
        - p[2]: log of Time constant  
        - p[3]: Exponent C
        """
        iwc = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc = torch.exp(-p[2]*p[3])
        if self.con:
            return torch.exp(-p[0])/(tc +(1.0-p[1])*iwc)*(tc+iwc)
        else:
            return torch.exp( p[0])*(tc +(1.0-p[1])*iwc)/(tc+iwc)      

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1
        ind_tau = 2
        ind_c = 3
       
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        mvec_tmp[ind_c] = torch.clamp(mvec[ind_c], self.clim.min(), self.clim.max())
        return mvec_tmp

class Pelton(Pelton_res_f):
    """Alias for Pelton_res_f with a simplified name."""
    pass

class ColeCole_f():
    def __init__(self, freq=None, res=False,
            conlim= [1e-5,1e2],
            chglim= [1e-3, 0.9],
            taulim= [1e-8, 1e4],
            clim= [0.2, 0.8]
                ):
        self.freq =  TorchHelper.to_tensor_c(freq)if freq is not None else None
        self.res = res
        self.reslim = TorchHelper.to_tensor_r(np.log(conlim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.clim = TorchHelper.to_tensor_r(clim)

    def show_equation(self):
        display(Markdown(r"""
### Cole-Cole Conductivity Model

$$
\sigma(\omega)=\sigma_\infty \left(1- \dfrac{\eta}{1+(i\omega \tau)^c}\right)
$$

- $\sigma_\infty$: Conductivity at high frequency ($\Omega\,$m)
- $\eta$: Chargeability (dimensionless)  
- $\tau$: Time constant (s)  
- $c$: Exponent C
"""))

    def f(self,p):
        """
        - p` (`torch.Tensor`): Model parameters in real number
        - p[0]: log of Conductivity at high frequency
        - p[1]: Chargeability  
        - p[2]: log of Time constant  
        - p[3]: Exponent C
        """
        iwc = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc = torch.exp(-p[2]*p[3])
        if self.res:
            return torch.exp(-p[0])/((1.0-p[1])*tc +iwc)*(tc+iwc)
        else:
            return torch.exp( p[0])*((1.0-p[1])*tc +iwc)/(tc+iwc)
        
    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()       
        mvec_tmp[0] = torch.clamp(mvec[0], self.reslim.min(), self.reslim.max())
        mvec_tmp[1] = torch.clamp(mvec[1], self.chglim.min(), self.chglim.max())
        mvec_tmp[2] = torch.clamp(mvec[2], self.taulim.min(), self.taulim.max())
        mvec_tmp[3] = torch.clamp(mvec[3], self.clim.min(), self.clim.max())
        return mvec_tmp

class ColeCole(ColeCole_f):
    """Alias for ColeCole_f with a simplified name."""
    pass

class DDR_f():
    def __init__(self,
            freq=None,
            times=None, tstep=None, con=False,
            taus=None,
            reslim= [1e-2,1e5],
            chglim= [0, 0.9],
            taulim= [1e-5, 1e1],
            taulimspc = [2,10],
                ):
        self.times = TorchHelper.to_tensor_r(times) if times is not None else None
        self.tstep = TorchHelper.to_tensor_r(tstep) if tstep is not None else None
        self.freq = TorchHelper.to_tensor_c(freq) if freq is not None else None
        self.taus = TorchHelper.to_tensor_c(taus) if taus is not None else None
        self.ntau = len(taus) if taus is not None else None
        self.con = con
        self.reslim = TorchHelper.to_tensor_r(np.log(reslim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.taulimspc = TorchHelper.to_tensor_r(np.log(taulimspc))

    def show_equation(self):
        display(Markdown(r"""
### Debye Decomposition Resistivity Model in frequency domain
$$
\rho(\omega)=\rho_0 \left[1-\sum_{j=1}^n \eta_j \left(1- \dfrac{1}{1+i\omega\tau_j}\right)\right]
$$

- $\rho_0$: Resistivity at low frequency ($\Omega\,$m)
- $\eta_j$: Chargeabilities (dimensionless)  
- $\tau_j$: Time constants (s)  
- $n$: Total number of relaxation
"""))

    def f(self, p):
        """
        - p` (`torch.Tensor`): Model parameters in real number
        - p[0]: log of Resistivity at low frequency
        - p[1:1+ntau]: chargeabilities
        """
        rho0 = torch.exp(p[0])
        etas = p[1:1 + self.ntau].to(dtype=torch.cfloat)
        omega = 2.0 * torch.pi * self.freq
        omega = omega.view(-1, 1)  # shape: [nfreq, 1]
        taus = self.taus.view(1, -1)  # shape: [1, ntau]
        etas = etas.view(1, -1)  # shape: [1, ntau]
        iwt = 1.0j * omega * taus  # shape: [nfreq, ntau]
        term =  etas / (1.0 + iwt)  # shape: [nfreq, ntau]
        if self.con:
            return 1.0/rho0 / (1.0 -etas.sum(dim=1)+ term.sum(dim=1))
        else:
            return rho0 * (1.0 -etas.sum(dim=1)+ term.sum(dim=1))  # shape: [nfreq]

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1+np.arange(self.ntau)
     
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        # mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], min=0, max=None)
        # mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        ax = torch.dot(a, x)
        if ax > b:
            proj_x = x + a * ((b - ax) / torch.dot(a, a))
        else:
            proj_x = x
        return proj_x

    def plot_etas(self, mvec, ax=None, **kwargs):
        if ax is None: 
            fig, ax = plt.subplots(1, 1, figsize=(5,3))
        etas = TorchHelper.to_numpy_r(mvec[1:])
        ax.semilogx(self.taus, etas, **kwargs)
        ax.set_xlabel(r"$\tau_k$ [s]")
        ax.set_ylabel(r"$\eta_k$")
        return ax

    def plot_etas_cum(self, mvec, ax=None, **kwargs):
        if ax is None: 
            fig, ax = plt.subplots(1, 1, figsize=(5,3))
        etas = TorchHelper.to_numpy_r(mvec[1:])
        ax.semilogx(self.taus, np.cumsum(etas), **kwargs)
        ax.set_xlabel(r"$\tau_k$ [s]")
        ax.set_ylabel(r"$\Sigma\!_{j,k}\,\eta_j$")
        return ax

class DDR_Sum_Ser_f(DDR_f):
    """Alias for DDR_f with a specific name."""
    pass

class DDR_t():
    def __init__(self,
            times=None, tstep=None, taus=None,
            reslim= [1e-2,1e5],
            chglim= [1e-3, 0.9],
            taulim= [1e-5, 1e1],
            taulimspc = [2,10],
                ):
        assert np.all(times >= -eps), "Times must be greater than or equal to 0"
        if len(times) > 1:
            assert np.all(np.diff(times) >= -eps), "Time values must be in ascending order."
        self.times = TorchHelper.to_tensor_r(times) if times is not None else None
        self.tstep = TorchHelper.to_tensor_r(tstep) if tstep is not None else None
        self.taus = TorchHelper.to_tensor_r(taus) if taus is not None else None
        self.ntau = len(taus) if taus is not None else None
        self.reslim = TorchHelper.to_tensor_r(np.log(reslim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.taulimspc = TorchHelper.to_tensor_r(np.log(taulimspc))

    def show_equation(self):
        display(Markdown(r"""
### Debye Decomposition Resistivity Model in time domain

$$
\rho(t)=\rho_0 \left[ \left(1 -\sum_{j=1}^n \eta_j \right) \delta(t)+ \sum_{j=1}^n \dfrac{\eta_j}{\tau_j}e^{\frac{-t}{\tau_j}}\right]
$$

- $\rho_0$: Resistivity at low frequency ($\Omega\,$m)
- $\eta_j$: Chargeabilities (dimensionless)  
- $\tau_j$: Time constants (s)  
- $n$: Total number of relaxation
"""))

    def t(self, p, tstep=None):
        """
        - p` (`torch.Tensor`): Model parameters in real number
        - p[0]: log of Resistivity at low frequency
        - p[1:1+ntau]: chargeabilities
        """
        if tstep is not None:
            self.tstep = TorchHelper.to_tensor_r(tstep)

        rho0 = torch.exp(p[0])
        etas = p[1:1 + self.ntau]
        ind_0 = torch.where(self.times == 0)[0]
        times = self.times.view(-1, 1)  # shape: [ntime, 1]
        # taus = taus.view(1, -1)  # shape: [1, ntau]
        taus = self.taus.view(1, -1)  # shape: [1, ntau]
        etas = etas.view(1, -1)  # shape: [1, ntau]
        term = etas/taus*torch.exp(-times/taus)  # shape: [ntime, ntau]
        term_sum = term.sum(dim=1)  # shape: [ntime]
        term_sum[ind_0] = 1.0-etas.sum(dim=1)  # Set the value at t=0 to 1 - sum(etas)
        if self.tstep is not None:
            ind_pos = torch.where(self.times > 0)
            term_sum[ind_pos] *= self.tstep
            return rho0 * (term_sum)  # shape: [tau]
        else:
            return rho0 * (term_sum)  # shape: [tau]
 
    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1+np.arange(self.ntau)
     
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg],-torch.ones(self.ntau), self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        ax = torch.dot(a, x)
        if ax > b:
            proj_x = x + a * ((b - ax) / torch.dot(a, a))
        else:
            proj_x = x
        return proj_x

class Debye_Sum_Ser_t(DDR_t):
    """Alias for Debye_sum_t with a specific name."""
    pass

class DDC_f():
    def __init__(self,
            freq=None, taus=None, res=False,
            conlim= [1e-5,1e2],
            chglim= [0, 0.9],
            taulim= [1e-5, 1e1],
            taulimspc = [2,10],
                ):
        self.freq = TorchHelper.to_tensor_c(freq) if freq is not None else None
        self.taus = TorchHelper.to_tensor_c(taus) if taus is not None else None
        self.ntau = len(taus) if taus is not None else None
        self.res= res
        self.conlim = TorchHelper.to_tensor_r(np.log(conlim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.taulimspc = TorchHelper.to_tensor_r(np.log(taulimspc))

    def show_equation(self):
        display(Markdown(r"""
### Debye Decomposition Conductivity Model in frequency domain

$$
\sigma(\omega)=\sigma_\infty\left(1- \sum_{j=1}^n\dfrac{\eta_j}{1+i\omega\tau_j}\right)
$$

- $\sigma_\infty$: Conductivity at high frequency ($\Omega\,$m)                         
- $\eta_j$: Chargeabilities (dimensionless)  
- $\tau_j$: Time constants (s)  
- $n$: Total number of relaxation
"""))

    def f(self, p):
        """
        - p` (`torch.Tensor`): Model parameters in real number
        - p[0]: log of Conductivity at high frequency 
        - p[1:1+ntau]: chargeabilities
        """
        con8 = torch.exp(p[0])
        etas = p[1:1 + self.ntau].to(dtype=torch.cfloat)
        omega = 2.0 * torch.pi * self.freq
        omega = omega.view(-1, 1)  # shape: [nfreq, 1]
        taus = self.taus.view(1, -1)  # shape: [1, ntau]
        etas = etas.view(1, -1)  # shape: [1, ntau]
        iwt = 1.0j * omega * taus  # shape: [nfreq, ntau]
        term = etas / (1.0 + iwt)  # shape: [nfreq, ntau]
        if self.res:
            return 1.0/con8 / (1.0 - term.sum(dim=1))
        else:
            return con8 * (1.0 - term.sum(dim=1))  # shape: [nfreq]

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_con = 0
        ind_chg = 1+np.arange(self.ntau)
        mvec_tmp[ind_con] = torch.clamp(mvec[ind_con], self.conlim.min(), self.conlim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg],-torch.ones(self.ntau), self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        ax = torch.dot(a, x)
        if ax > b:
            proj_x = x + a * ((b - ax) / torch.dot(a, a))
        else:
            proj_x = x
        return proj_x

class DDC_t():
    def __init__(self,
            times=None, tstep=None, taus=None,
            conlim= [1e-5,1e3],
            chglim= [1e-3, 0.9],
            taulim= [1e-5, 1e1],
            taulimspc = [2,10],
                ):
        assert np.all(times >= -eps), "Times must be greater than or equal to 0"
        if len(times) > 1:
            assert np.all(np.diff(times) >= -eps), "Time values must be in ascending order."
        self.times = TorchHelper.to_tensor_r(times) if times is not None else None
        self.tstep = TorchHelper.to_tensor_r(tstep) if tstep is not None else None
        self.taus = TorchHelper.to_tensor_r(taus) if taus is not None else None
        self.ntau = len(taus) if taus is not None else None
        self.conlim = TorchHelper.to_tensor_r(np.log(conlim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.taulimspc = TorchHelper.to_tensor_r(np.log(taulimspc))

    def show_equation(self):
        display(Markdown(r"""
### Debye Decomposition Conductivity Model in frequency domain

$$
\sigma(t)=\sigma_\infty \left[ \delta(t)- \sum_{j=1}^n \dfrac{\eta_j}{\tau_j}e^{\frac{-t}{\tau_j}}\right]
$$

- $\sigma_\infty$: Conductivity at high frequency ($\Omega\,$m)                         
- $\eta_j$: Chargeabilities (dimensionless)  
- $\tau_j$: Time constants (s)  
- $n$: Total number of relaxation
"""))

    def t(self, p, tstep=None):
        """
        - p` (`torch.Tensor`): Model parameters in real number
        - p[0]: log of Conductivity at high frequency 
        - p[1:1+ntau]: chargeabilities
        """
        if tstep is not None:
            self.tstep = TorchHelper.to_tensor_r(tstep)

        con8 = torch.exp(p[0])
        etas = p[1:1 + self.ntau]
        ind_0 = torch.where(self.times == 0)[0]
        times = self.times.view(-1, 1)  # shape: [ntime, 1]
        taus = self.taus.view(1, -1)  # shape: [1, ntau]
        etas = etas.view(1, -1)  # shape: [1, ntau]
        term = etas/taus*torch.exp(-times/taus)  # shape: [ntime, ntau]
        term_sum = term.sum(dim=1)  # shape: [ntime]
        term_sum[ind_0] = 1  # Set the value at t=0 to 1 - sum(etas)
        if self.tstep is not None:
            ind_pos = torch.where(self.times > 0)
            term_sum[ind_pos] *= self.tstep
        return con8 * (term_sum)  # shape: [tau]

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_con = 0
        ind_chg = 1+np.arange(self.ntau)
    
        mvec_tmp[ind_con] = torch.clamp(mvec[ind_con], self.conlim.min(), self.conlim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg],-torch.ones(self.ntau), self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        ax = torch.dot(a, x)
        if ax > b:
            proj_x = x + a * ((b - ax) / torch.dot(a, a))
        else:
            proj_x = x
        return proj_x

class DDP_r_f():
    def __init__(self,
            freq=None, con=False, taus=None,
            reslim= [1e-2,1e5],
            chglim= [0, 0.9],
            taulim= [1e-5, 1e1],
            taulimspc = [2,10],
                ):
        self.freq = TorchHelper.to_tensor_c(freq) if freq is not None else None
        self.taus = TorchHelper.to_tensor_c(taus) if taus is not None else None
        self.ntau = len(taus) if taus is not None else None
        self.con = con
        self.reslim = TorchHelper.to_tensor_r(np.log(reslim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.taulimspc = TorchHelper.to_tensor_r(np.log(taulimspc))

    def show_equation(self):
        display(Markdown(r"""
### Debye Decomposition using relaxation time constants of Maximum Phase in resistivity form in frequency domain
$$
\rho(\omega)=\rho_0 \left[1-\sum_{j=1}^n \eta_j \left(1- \dfrac{1}{1+(1-\eta_j)^{-\frac{1}{2}}i\omega\tau_j}\right)\right]
$$

- $\rho_0$: Resistivity at low frequency ($\Omega\,$m)
- $\eta_j$: Chargeabilities (dimensionless)  
- $\tau_j$: Time constants (s)  
- $n$: Total number of relaxation
"""))

    def f(self, p):
        """
        p: torch.Tensor
        p[0]  = log(rho0)
        p[1:] = unconstrained chargeabilities parameters (mapped to (0,1) below)
        """
        device = self.freq.device
        real_dtype = self.freq.dtype  # e.g., torch.float32 or float64

        rho0 = torch.exp(p[0]).to(device=device, dtype=real_dtype)

        # Start real. If upstream passed complex, take the real part (keeps autograd).
        etas_1d = p[1:1 + self.ntau].to(device=device, dtype=real_dtype)
        if torch.is_complex(etas_1d):
            etas_1d = etas_1d.real


        # If needed, map to (0,1) for stability; remove sigmoid if you already constrain etas.
        # etas_1d = torch.sigmoid(etas_1d)

        etas = etas_1d.view(1, -1)  # [1, ntau], real
        omega = (2.0 * torch.pi * self.freq).view(-1, 1).to(device=device, dtype=real_dtype)  # [nfreq, 1]
        taus  = self.taus.view(1, -1).to(device=device, dtype=real_dtype)                      # [1, ntau]

        eps = torch.finfo(real_dtype).eps
        inv_sqrt = torch.rsqrt((1.0 - etas).clamp_min(eps))  # real & safe: [1, ntau]
        # (1 - η)^(-1/2) * i ω τ  ==  i ω τ / sqrt(1-η)
        iwt = 1j * omega * taus * inv_sqrt                   # [nfreq, ntau], complex

        term = etas / (1.0 + iwt)                            # [nfreq, ntau], complex

        base = 1.0 - etas_1d.sum()                           # scalar (real)
        accum = base + term.sum(dim=1)                       # [nfreq], complex

        if self.con:  # conductivity
            out = (1.0 / rho0) / accum
        else:         # resistivity
            out = rho0 * accum

        return out.to(torch.cfloat)

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1+np.arange(self.ntau)
     
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg],-torch.ones(self.ntau), self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        ax = torch.dot(a, x)
        if ax > b:
            proj_x = x + a * ((b - ax) / torch.dot(a, a))
        else:
            proj_x = x
        return proj_x

class Pelton_res_f_two():
    def __init__(self, freq=None,
            reslim= [1e-2,1e5],
            chglim= [1e-3, 0.9],
            taulim= [1e-8, 1e4],
            clim= [0.2, 0.8]
                ):
        self.freq = torch.tensor(freq, dtype=torch.cfloat) if freq is not None else None
        self.reslim = torch.tensor(np.log(reslim))
        self.chglim = torch.tensor(chglim)
        self.taulim = torch.tensor(np.log(taulim))
        self.clim = torch.tensor(clim)
    
    def f(self,p):
        """
        Pelton two relaxation resistivity model
        made easy for PyTorch Auto Diffentiation.
        p[0] : log(res0)
        p[1] : eta
        p[2] : log(tau1)
        p[3] : c1
        p[4] : log(tau2)
        p[5] : c2
        """
        iwc1 = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc1 = torch.exp(-p[2]*p[3])
        iwc2 = (1j * 2. * torch.pi * self.freq  ) ** p[5]
        tc2 = torch.exp(-p[4]*p[5])
        f = torch.exp(p[0])*(tc1 +(1.0-p[1])*iwc1)*(tc2)/(tc1+iwc1)/(tc2+iwc2)
        return f
    
    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1
        ind_tau = [2,4]
        ind_c = [3,5] 
    
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        mvec_tmp[ind_c] = torch.clamp(mvec[ind_c], self.clim.min(), self.clim.max())
        return mvec_tmp

class Pelton_res_f_dual():
    def __init__(self, freq=None,
        reslim= [1e-2,1e5],
        chglim= [1e-3, 0.9],
        taulim= [1e-8, 1e4],
        clim= [0.2, 0.8]
            ):
        self.freq = torch.tensor(freq, dtype=torch.cfloat) if freq is not None else None
        self.reslim = torch.tensor(np.log(reslim))
        self.chglim = torch.tensor(chglim)
        self.taulim = torch.tensor(np.log(taulim))
        self.clim = torch.tensor(clim)

    def f(self,p):
        """
        Pelton dual cole-cole resistivity model
        made easy for PyTorch Auto Diffentiation.
        p[0] : log(res0)
        p[1] : eta1
        p[2] : log(tau1)
        p[3] : c1
        p[4] : eta2
        p[5] : log(tau2)
        p[6] : c2
        """
        iwc1 = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc1 = torch.exp(-p[2]*p[3])
        iwc2 = (1j * 2. * torch.pi * self.freq  ) ** p[6]
        tc2 = torch.exp(-p[5]*p[6])
        f = torch.exp(p[0])*(tc1 +(1.0-p[1])*iwc1)/(tc1+iwc1)*(tc2 +(1.0-p[4])*iwc2)/(tc2+iwc2)
        return f

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = [1,4]
        ind_tau = [2,5]
        ind_c = [3,6] 
    
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        mvec_tmp[ind_c] = torch.clamp(mvec[ind_c], self.clim.min(), self.clim.max())
        return mvec_tmp

class BaseSimulation:
    eps = torch.finfo(torch.float32).eps
    @abstractmethod
    def dpred(self,m):
        pass
    @abstractmethod
    def J(self,m):
        pass
    @abstractmethod
    def project_convex_set(self,m):
        pass

class InducedPolarizationSimulation(BaseSimulation):
    AVAILABLE_MODES = ['tdip_t', 'tdip_f', 'sip_t', 'sip']

    eps = torch.finfo(torch.float32).eps
    def __init__(self, 
                 ip_model=None,
                 mode=None,
                 times=None,
                 basefreq=None,
                 window_mat=None,
                 windows_strt=None,
                 windows_end=None,
                 log2min=-6,
                 log2max=6,
                 ):
        """
        Induced Polarization Simulation.

        Parameters:
        - ip_model: input IP model for simulation
        - mode (str): One of ['tdip_t', 'tdip_f', 'sip_t', 'sip']
        - times: Time values (1D list or tensor)
        - basefreq: Frequency base (1D list or tensor)
        - window_mat: Matrix used for windowing the response
        - windows_strt, windows_end: (Optional) Start and end times for windows
        """
        #  Validate the mode
        if mode is not None and mode not in self.AVAILABLE_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.AVAILABLE_MODES}")

        self.ip_model = ip_model
        self.mode=mode
        self.times = TorchHelper.to_tensor_r(times) if times is not None else None
        self.basefreq = TorchHelper.to_tensor_r(basefreq) if basefreq is not None else None
        self.window_mat = TorchHelper.to_tensor_r(window_mat) if window_mat is not None else None 
        self.windows_strt = TorchHelper.to_tensor_r(windows_strt) if windows_strt is not None else None
        self.windows_end = TorchHelper.to_tensor_r(windows_end) if windows_end is not None else None
        self.windows_width = TorchHelper.to_tensor_r(windows_end-windows_strt) if windows_strt is not None else None
        self.log2min = log2min
        self.log2max = log2max
   
    def count_data_windows(self, times):
        nwindows = len(self.windows_strt)
        count_data = torch.zeros(nwindows)
        for i in torch.arange(nwindows):
            start = self.windows_strt[i]
            end = self.windows_end[i]
            ind_time = (times >= start) & (times <= end)
            count_data[i] = ind_time.sum()
        return count_data

    def get_freq_windowmat(self,tau, max=22, ign_0=False):
        log2max = self.log2max
        log2min = self.log2min
        freqend = ((1 / tau) * 2 ** log2max).item()
        freqstep = ((1 / tau) * 2 ** log2min).item()
        freq = torch.arange(0, freqend, freqstep)
        times = torch.arange(0, 1/freqstep,1/freqend)

        count = self.count_data_windows(times)
        # while count[self.windows_end==self.windows_end.max()] <2:
        #     # print(count)
        #     log2min -= 1
        #     freqstep = ((1/tau)*2**log2min).item()
        #     if log2max-log2min >= max:
        #         print('some windows are too narrow')
        #         break
        #     freq = torch.arange(0,freqend,freqstep)
        #     times = torch.arange(0, 1/freqstep,1/freqend)
        #     count = self.count_data_windows(times)

        while times.max() < self.windows_end.max():
            # print(count)
            log2min -= 1
            freqstep = ((1/tau)*2**log2min).item()
            if log2max-log2min >= max:
                print('some windows are too narrow')
                break
            freq = torch.arange(0,freqend,freqstep)
            times = torch.arange(0, 1/freqstep,1/freqend)
            # freq = torch.fft.fftfreq(len(freq), d=1/freqend)
            count = self.count_data_windows(times)
        
        ind_narrow = self.windows_width == self.windows_width.min()
        if ind_narrow.sum() >= 2:
            # print(torch.where(ind_narrow)[0])
            # print(torch.where(ind_narrow)[0][0].item())
            ind_narrow = torch.where(ind_narrow)[0][0].item()

        while count[ind_narrow] <2:
            # print(count)
            log2max += 1
            freqend = ((1/tau)*2**log2max).item()
            if log2max-log2min >= max:
                print('some windows are too narrow')
                break
            freq = torch.arange(0,freqend,freqstep)
            times = torch.arange(0, 1/freqstep,1/freqend)
            # freq = torch.fft.fftfreq(len(times), d=1/freqend)
            count = self.count_data_windows(times)

        self.times = times
        # freq = torch.fft.fftfreq(len(freq), d=1/freqend)

        self.ip_model.freq = freq
        self.get_window_matrix(times=times)
        
    def freq_symmetric(self,f):
        f_sym = torch.zeros_like(f, dtype=torch.cfloat)
        nfreq = len(f)
        if nfreq  %2 == 0:
            nfreq2 = nfreq // 2
            f_sym[:nfreq2] = f[:nfreq2]
            f_sym[nfreq2] =f[nfreq2].real
            f_sym[nfreq2+1:] = torch.flip(f[1:nfreq2].conj(), dims=[0])
        # Ensure symmetry at the Nyquist frequency (if even length)
        else:
            nfreq2 = nfreq // 2 
            f_sym[:nfreq2+1] = f[:nfreq2+1]
            f_sym[nfreq2+1:] = torch.flip(f[1:nfreq2+1].conj(), dims=[0])
        return f_sym

    def compute_fft(self,f):
        f_sym = self.freq_symmetric(f)
        t = torch.fft.ifft(f_sym) 
        return t

    def get_windows(self,windows_cen):
        # windows_cen = TorchHelper.to_tensor_r(windows_cen[windows_cen>0])
        windows_cen = TorchHelper.to_tensor_r(windows_cen)
        windows_strt = torch.zeros_like(windows_cen)
        windows_end  = torch.zeros_like(windows_cen)
        dt = torch.diff(windows_cen)
        windows_strt[1:] = windows_cen[:-1] + dt / 2
        windows_end[:-1] = windows_cen[1:] - dt / 2
        # windows_strt[0] = 10*eps
        windows_strt[0] = windows_cen[0] - dt[0] / 2
        windows_end[-1] = windows_cen[-1] + dt[-1] / 2
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        self.windows_width = windows_end - windows_strt

    def get_window_matrix(self, times=None, sum=False):
        if times is not None:
            self.times = TorchHelper.to_tensor_r(times)
        nwindows = len(self.windows_strt)
        window_matrix = torch.zeros((nwindows, len(self.times)))
        for i in range(nwindows):
            ind_time = (self.times >= self.windows_strt[i]) & (self.times <= self.windows_end[i])
            count = ind_time.sum()
            if count > 0:
                if sum:
                    window_matrix[i, ind_time] = 1.0
                else:
                    window_matrix[i, ind_time] = 1.0 / count
            else:
                print(f"Warning: No data points found for window {i+1} ({self.windows_strt[i]} to {self.windows_end[i]})")
        self.window_mat = window_matrix
        # window_matrix = torch.zeros((nwindows+1, len(self.times)))
        # window_matrix[0,0] = 1
        # for i in range(nwindows):
        #     ind_time = (self.times >= self.windows_strt[i]) & (self.times <= self.windows_end[i])
        #     if ind_time.sum() > 0:
        #         if sum:
        #             window_matrix[i+1, ind_time] = torch.ones(ind_time.sum())
        #         window_matrix[i+1, ind_time] = 1.0/(ind_time.sum())
        # self.window_mat = window_matrix

    def set_current_wave(self, basefreq=None, curr_duty=0.5):
        """
        Set the current wave for the simulation as rectangular wave form.
        """
        if basefreq is not None:
            self.basefreq = basefreq
        curr =  0.5*(1.0+signal.square(2*np.pi*(self.basefreq*self.times),duty=curr_duty))
        self.curr = TorchHelper.to_tensor_r(curr)
        self.curr_duty=  curr_duty

    def get_windows_matrix_curr(self, smp_freq, nlin, nlin_strt, basefreq=None) :
        """
        Get the windows matrix based on currenqt wave form.
        """
        if basefreq is not None:
            self.basefreq = basefreq
        rec_time = 1/self.basefreq
        time_step = 1/smp_freq
        windows_lin = (np.arange(nlin)+nlin_strt)*time_step 
        windows_log_strt = (nlin+nlin_strt)*time_step
        logstep = np.log10(windows_lin[-1]/windows_lin[-2])
        windows_log_end = self.curr_duty/self.basefreq
        windows_log = 10**np.arange(
            np.log10(windows_log_strt),
            np.log10(windows_log_end), logstep)
        windows_cen=np.r_[windows_lin, windows_log]
        nhalf = len(windows_cen)
        windows_cen = np.r_[windows_cen, self.curr_duty/self.basefreq + windows_cen]
        self.get_windows(windows_cen)

        self.windows_end[nhalf-1]  = self.curr_duty/self.basefreq
        self.windows_strt[nhalf] = self.windows_strt[0] + self.curr_duty/self.basefreq
        self.windows_end[-1] = 1/self.basefreq
        self.get_window_matrix(times=self.times)

    def fft_convolve(self, d, f):
        """
        Perform 1D linear convolution using FFT (like scipy.signal.fftconvolve).
        Assumes x and h are 1D tensors.
        Returns output of length (len(x) + len(h) - 1)
        """
        nd = d.shape[0] 
        nf = f.shape[0] 
        # Compute FFTs (real FFT for speed)
        D = torch.fft.rfft(d, n=nd+nf-1)
        F = torch.fft.rfft(f, n=nd+nf-1)
        # Element-wise multiplication
        DF = D * F
        # Inverse FFT to get back to time domain
        return  torch.fft.irfft(DF, n=nd+nf-1)
 
    def dpred(self,m):
        if self.mode=="tdip_t" :
            ip_t=self.ip_model.t(m)
            volt = self.fft_convolve(ip_t,self.curr)[: len(self.times)]
            return self.window_mat@volt

        if self.mode=="tdip_f":
            ip_f = self.ip_model.f(m)
            ip_t = torch.fft.ifft(ip_f).real
            volt = self.fft_convolve(ip_t,self.curr)[: len(self.times)]
            return self.window_mat@volt

        # if self.mode=="tdip_f":
        #     self.get_freq_windowmat(tau=torch.exp(m[2]))
        #     self.set_current_wave()
        #     ip_f = self.ip_model.f(m)
        #     ip_fsym = self.freq_symmetric(ip_f)
        #     ip_t = torch.fft.ifft(ip_fsym).real
        #     volt = self.fft_convolve(ip_t,self.curr)[: len(self.times)]
        #     return self.window_mat@volt
            # assert len(self.times) == len(self.ip_model.freq)
            # curr_f = torch.fft.fft(self.curr)
            # volt = torch.fft.ifft(ip_fsym*curr_f).real
            # volt = volt[: len(self.times)]
            # return self.window_mat@volt

        if self.mode=="sip_t":
            # self.get_freq_windowmat(tau=torch.exp(m[2]))
            f = self.ip_model.f(m)
            t = self.compute_fft(f)/(self.times[1]-self.times[0])
            t_real = t.real
            return self.window_mat@t_real
        
        if self.mode=="sip":
            f = self.ip_model.f(m)
            f_real = f.real
            f_imag = f.imag
            if self.window_mat is None:
                return torch.cat([f_real, f_imag])
            else:
                return torch.cat([self.window_mat@f_real, self.window_mat@f_imag])

    def J(self,m):
        return torch.autograd.functional.jacobian(self.dpred, m)   

    def J_prd(self,J):
        """"
        Retursn numpy arrays given the Jacobian matrix J in numpy format.
        1. J_pro: the projection of the eta vectors on the resistivity vector, normalized
        2. etas_norm: the norm of the eta vectors, normalized by the norm of Jacobian matrix
        """
        J_0 = J[:,0]
        J_0_norm = np.linalg.norm(J_0)
        J_prd = np.zeros(J.shape[1]-1)
        for i in range(J.shape[1]-1):
            J_i = J[:,i+1]
            J_i_norm = np.linalg.norm(J_i)
            J_prd[i] = np.dot(J_0, J_i)/J_i_norm/ J_0_norm
        return J_prd

    def Jvec(self, m, v):
        return torch.autograd.functional.jvp(self.dpred, m, v)

    def Jtvec(self, m, v):
        return torch.autograd.functional.vjp(self.dpred, m, v)

    def project_convex_set(self,m):
        return self.ip_model.clip_model(m).detach().requires_grad_(False)
    
    def plot_sip_model(self, model, res=True, ax=None, magphs=True, deg=True,**kwargs):
        model = TorchHelper.to_tensor_r(model)
        dpred = self.dpred(model)
        if magphs:
            axtmp = [None, None, None, ax[0], ax[1], None]
        else:
            axtmp = ax
        if res:
            axtmp = self.plot_sip_dpred_res(dpred, ax=axtmp, **kwargs)
        else:
            axtmp = self.plot_sip_dpred_con(dpred, ax=axtmp, **kwargs)
        if magphs:
            ax = [axtmp[3], axtmp[4]]
        else:
            ax = axtmp 
        return ax

    def plot_sip_dpred_res(self, dpred,  deg=True, ax=None, **kwargs):
        """
        Plot SIP given data of resistivity form
        dpred: torch.Tensor, transformed to numpy for plotting
        ax: sequence of matplotlib Axes (len>=5) or None
            ax[0]: real part in resistivity vs freq
            ax[1]: imag part in resistivity vs freq
            ax[2]: imag part in conductivity vs freq
            ax[3]: abs   in resistivity vs freq
            ax[4]: phase in resistivity vs freq
            ax[5]: Cole-Cole (imag vs real)
        """
        if ax is None:
            fig, ax = plt.subplots(3, 2, figsize=(11, 9))
            ax = ax.ravel()
        else:
            # accept single Axes, 2D array, list/tuple
            ax = np.asarray(ax, dtype=object).ravel()

        if ax.size < 6:
            raise ValueError(f"Need at least 6 axes (got {ax.size}).")
        dpred_np = TorchHelper.to_numpy_c(dpred)
        freq = TorchHelper.to_numpy_r(self.ip_model.freq)
        nfreq = freq.shape[0]

        sip_real = dpred_np[:nfreq]
        sip_imag = dpred_np[nfreq:nfreq*2]  # safer if dpred has extra stuff
        z = sip_real + 1j * sip_imag
        a = 1.0 / z
        sip_abs = np.abs(z)
        sip_phs = np.angle(z, deg=deg)
        if deg is False:
            sip_phs *= 1000 # convert to mrad
        # Frequency-domain plots
        freq_axes = [0, 1, 2, 3, 4]
        if ax[0] is not None:
            ax[0].semilogx(freq, sip_real, **kwargs)
            ax[0].set_ylabel(r"Re($\rho^*$)  ($\Omega$m)")
        if ax[1] is not None:
            ax[1].semilogx(freq, sip_imag, **kwargs)
            ax[1].set_ylabel(r"Im($\rho^*$)  ($\Omega$m)")
            ax[1].set_ylim(top=0)  # resistivity imaginary part is negative

        if ax[2] is not None:
            ax[2].semilogx(freq, a.imag, **kwargs)
            ax[2].set_ylabel(r"Im($\sigma^*$)  (S/m)")
            ax[2].set_ylim(bottom=0)  # conductivity imaginary part is positive
        if ax[3] is not None:
            ax[3].semilogx(freq, sip_abs, **kwargs)
            ax[3].set_ylabel(r"Amplitude ($\Omega$m)")
        if ax[4] is not None:
            ax[4].semilogx(freq, sip_phs, **kwargs)
            if deg:
                ax[4].set_ylabel("Phase (deg)")
            else:
                ax[4].set_ylabel("Phase (mrad)")

        for i in freq_axes:
            if ax[i] is not None:
                ax[i].set_xlabel("Frequency (Hz)")

        # Cole–Cole
        if ax[5] is not None:
            ax[5].plot(sip_real, sip_imag, **kwargs)
            ax[5].set_xlabel(r"Re($\rho^*$)  ($\Omega$m)")
            ax[5].set_ylabel(r"Im($\rho^*$)  ($\Omega$m)")
            ax[5].axis("equal")  # optional, often nice
        return ax

    def plot_sip_dpred_con(self, dpred, deg=True, ax=None, **kwargs):
        """
        Plot SIP given data of conductivity form
        dpred: torch.Tensor, transformed to numpy for plotting
        ax: sequence of matplotlib Axes (len>=5) or None
            ax[0]: real part in conductivity vs freq
            ax[1]: imag part in conductivity vs freq
            ax[2]: imag part in resistivity vs freq
            ax[3]: abs   in conductivity vs freq
            ax[4]: phase in conductivity vs freq
            ax[5]: Cole-Cole (imag vs real)
        """
        if ax is None:
            fig, ax = plt.subplots(3, 2, figsize=(11, 9))
            ax = ax.ravel()
        else:
            # accept single Axes, 2D array, list/tuple
            ax = np.asarray(ax, dtype=object).ravel()

        if ax.size < 6:
            raise ValueError(f"Need at least 6 axes (got {ax.size}).")
        dpred_np = TorchHelper.to_numpy_c(dpred)
        freq = TorchHelper.to_numpy_r(self.ip_model.freq)
        nfreq = freq.shape[0]

        sip_real = dpred_np[:nfreq]
        sip_imag = dpred_np[nfreq:nfreq*2]  # safer if dpred has extra stuff
        a = sip_real + 1j * sip_imag
        z = 1.0 / a
        sip_abs = np.abs(a)
        sip_phs = np.angle(z, deg=deg)
        if deg is False:
            sip_phs *= 1000 # convert to mrad
        # Frequency-domain plots
        freq_axes = [0, 1, 2, 3, 4]
        if ax[0] is not None:
            ax[0].semilogx(freq, sip_real, **kwargs)
            ax[0].set_ylabel(r"Re($\sigma^*$)  (S/m)")
        if ax[1] is not None:
            ax[1].semilogx(freq, sip_imag, **kwargs)
            ax[1].set_ylabel(r"Im($\sigma^*$)  (S/m)")
            ax[1].set_ylim(bottom=0)  # conductivity imaginary part is positive
        if ax[2] is not None:
            ax[2].semilogx(freq, z.imag, **kwargs)
            ax[2].set_ylabel(r"Im($\rho^*$)  ($\Omega\cdot$m)")
            ax[2].set_ylim(top=0)  # resistivity imaginary part is negative

        if ax[3] is not None:
            ax[3].semilogx(freq, sip_abs, **kwargs)
            ax[3].set_ylabel(r"|$\sigma^*$|  (S/m)")
        if ax[4] is not None:
            ax[4].semilogx(freq, sip_phs, **kwargs)
            if deg:
                ax[4].set_ylabel("Phase (deg)")
            else:
                ax[4].set_ylabel("Phase (mrad)")

        for i in freq_axes:
            if ax[i] is not None:
                ax[i].set_xlabel("Frequency (Hz)")

        # Cole–Cole
        if ax[5] is not None:
            ax[5].plot(sip_real, sip_imag, **kwargs)
            ax[5].set_xlabel(r"Re($\sigma^*$)  (S/m)")
            ax[5].set_ylabel(r"Im($\sigma^*$)  (S/m)")
            ax[5].axis("equal")  # optional, often nice
        return ax
    
class Optimization():  # Inherits from BaseSimulation
    def __init__(self,
                sim,
                dobs=None, Jmat=None, Wd=None, 
                beta=None, Ws=None, Wx=None,
                alphas=1, alphax=1, Ws_threshold=1e-6
                ):
        self.sim = sim  # Composition: opt_tmp has a simulation
        self.dobs= dobs
        self.Jmat = Jmat
        self.Wd = Wd
        self.beta=beta
        self.Ws = Ws
        self.Ws_threshold = Ws_threshold
        self.Wx = Wx
        self.alphas = alphas
        self.alphax = alphax
    
    def dpred(self, m):
        return self.sim.dpred(m)  # Calls InducedPolarization's dpred()

    def J(self, m):
        self.Jmat = self.sim.J(m)
        return self.Jmat  # Calls InducedPolarization's J()
    
    def Jvec(self, m, v):
        return self.sim.Jvec(m, v)
    
    def Jtvec(self, m, v):
        return self.sim.Jtvec(m, v)
    
    def project_convex_set(self,m):
        return self.sim.project_convex_set(m)

    def get_Wd(self,ratio=0.10, plateau=0, sparse=False):
        dobs_clone = self.dobs.clone().detach()
        noise_floor = plateau * torch.ones_like(dobs_clone)
        std = torch.sqrt(noise_floor**2 + (ratio * torch.abs(dobs_clone))**2)
        self.Wd =torch.diag(1 / std.flatten())
        if sparse:
            self.Wd = self.Wd.to_sparse()

    def get_Ws(self, mvec, sparse=False):
        self.Ws = torch.eye(mvec.shape[0])
        if sparse:
            self.Ws = self.Ws.to_sparse()
    
    def loss_func_L2(self,m, beta, m_ref=None):
        r = self.dpred(m)-self.dobs
        r = self.Wd @ r
        phid = torch.sum(r**2)
        phim = 0
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim = 0.5 * self.alphas*torch.sum(rms**2)
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += 0.5 * self.alphax*torch.sum(rmx**2) 
        return phid+beta*phim, phid, phim 

    def loss_func_L2reg(self, m, m_ref=None):
        dpred_m = self.dpred(m)
        r = dpred_m-self.dobs
        r = self.Wd @ r
        phid = torch.sum(r**2)
        phim = 0
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim = 0.5 * self.alphas*torch.sum(rms**2)
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += 0.5 * self.alphax*torch.sum(rmx**2) 
        return phid+self.beta*phim, phid, phim, dpred_m

    def loss_func_L2reg_gH(self,m,update_Wsen=False,m_ref=None):
        Jmat = self.J(m)
        if update_Wsen:
            self.update_Ws(self.Wd@Jmat)
        dpred_m = self.dpred(m)            
        rd = self.Wd @ (dpred_m -self.dobs)
        phid = torch.sum(rd**2)
        H = Jmat.T @ self.Wd.T@ self.Wd@Jmat
        phim = 0
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim += 0.5 * self.alphas*torch.sum(rms**2)
            H += self.beta * self.alphas * self.Ws.T@self.Ws
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += 0.5 * self.alphax*torch.sum(rmx**2)
            H += self.beta * self.alphax * self.Wx.T @ self.Wx        
        f= phid+self.beta*phim
        f.backward()
        g=m.grad 
        return f, phid, phim, dpred_m, g, H

    def loss_func_L1reg(self, m, m_ref=None):
        dpred_m = self.dpred(m)
        r = dpred_m-self.dobs
        r = self.Wd @ r
        phid = torch.sum(r**2)
        phim = 0
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim += self.alphas*torch.sum(rms**2)
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += self.alphax*torch.sum(rmx**2)
        return phid+self.beta*phim, phid, phim, dpred_m
   
    def loss_func_L1reg_gH(self,m,update_Wsen=False,m_ref=None):
        dpred_m = self.dpred(m)            
        Jmat = self.J(m)
        if update_Wsen:
            self.update_Ws(self.Wd@Jmat)
        rd = self.Wd @ (dpred_m -self.dobs)
        phid = torch.sum(rd**2)
        H = Jmat.T @ self.Wd.T@ self.Wd@Jmat
        phim = 0
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim += self.alphas*torch.sum(abs(rms))
            H += self.beta * self.alphas * self.Ws.T@self.Ws
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += self.alphax*torch.sum(abs(rmx))
            H += self.beta * self.alphax * self.Wx.T@self.Wx
        f = phid + self.beta*phim
        # Compute the gradient of f
        f.backward()
        g = m.grad 
        # Compute the Hessian
        return f, phid, phim, dpred_m, g, H

    def loss_func_Jacobian_proj(self,m,m_ref=None):
        dpred_m = self.dpred(m)            
        rd = self.Wd @ (dpred_m -self.dobs)
        phid = torch.sum(rd**2)
        phim = 0
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim += self.alphas*torch.sum(abs(rms))
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += self.alphax*torch.sum(abs(rmx))
        f = phid + self.beta*phim
        return f, phid, phim, dpred_m

    def loss_func_Jacobian_proj_gh(self,m,update_Wsen=False,m_ref=None):
        dpred_m = self.dpred(m)            
        Jmat = self.J(m)
        WdJ = self.WdJ_proj(self.Wd @ Jmat, ind_0=0)
        if update_Wsen:
            self.update_Ws(WdJ)
        rd = self.Wd @ (dpred_m -self.dobs)
        phid = torch.sum(rd**2)
        H = WdJ.T@ WdJ
        g = WdJ.T@ rd
        phim = 0
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim += self.alphas*torch.sum(abs(rms))
            H += self.beta * self.alphas * self.Ws.T@self.Ws
            g += self.beta * self.alphas * self.Ws.T @ rms
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += self.alphax*torch.sum(abs(rmx))
            H += self.beta * self.alphax * self.Wx.T @ self.Wx
            g += self.beta * self.alphax * self.Wx.T @ rmx
        f = phid + self.beta*phim
        return f, phid, phim, dpred_m, g, H

    def WdJ_proj(self, WdJ, ind_0=0, ind_sel=None, beta=None):

        Nd, Nm = WdJ.shape

        # default set: all but rho0
        if ind_sel is None:
            ind_sel = [i for i in range(Nm) if i != ind_0]
        ind_sel = torch.tensor(ind_sel, dtype=torch.long, device=WdJ.device)

        # build u (column vector)
        WdJ0 = WdJ[:, ind_0]
        nrm = torch.norm(WdJ0)
        if nrm.item() == 0:
            raise ValueError("||WdJ[:, ind_0]|| == 0; cannot form projection direction.")
        u = (WdJ0 / nrm).view(Nd, 1)

        # columns to project
        Je = WdJ[:, ind_sel]                         # (Nd, k)
        a = torch.matmul(u.T, Je).view(-1)                       # (k,)
        if beta is not None:
            beta = torch.as_tensor(beta, dtype=WdJ.dtype, device=WdJ.device).view(-1)
            if beta.numel() != a.numel():
                raise ValueError("beta must have length equal to len(ind_sel).")
            a = a * beta                             # scale projections per column

        # rank-1 projection: Je_perp = Je - u @ (u.T @ Je)
        Je_perp = Je - torch.matmul(u, a.unsqueeze(0))  # (Nd, k)

        # assemble result
        WdJ_out = WdJ.clone()
        WdJ_out[:, ind_sel] = Je_perp
        WdJ_out[:, ind_0] = WdJ0
        return WdJ_out 

    def BetaEstimate_byEig(self,m, beta0_ratio=1.0, 
                eig_tol=eps,l1reg=False, norm=True,update_Wsen=False):
        mvec = m.clone().detach()
        J = self.J(mvec)

        if update_Wsen:
            self.update_Ws(self.Wd@J)    

        # Prj_m = self.Prj_m  # Use `Proj_m` to map the model space

        # Effective data misfit term with projection matrix
        A_data = 0.5* J.T @ self.Wd.T @ self.Wd @ J 
        
        # Effective regularization term with projection matrix
        # A_reg = alphax* Prj_m.T @ Wx.T @ Wx @ Prj_m
        A_reg = torch.zeros_like(A_data)
        if self.Wx is not None:
            if l1reg:
                diag = torch.diag(self.Wx.T @ self.Wx)
                A_reg += self.alphax *torch.diag(diag**0.5)
            else:
                A_reg += 0.5*self.alphax * (self.Wx.T @ self.Wx)
        if self.Ws is not None:
            if l1reg:
                A_reg += self.alphas * self.Ws
            else:
                A_reg += 0.5*self.alphas * (self.Ws.T @ self.Ws)

        if norm:
            lambda_d = torch.linalg.norm(A_data, ord=2)  # Spectral norm ≈ largest eigval
            lambda_r = torch.linalg.norm(A_reg, ord=-2)  # Smallest eigval approx (not accurate, but fast)
        else:
            eig_data = torch.linalg.eigvalsh(A_data)
            eig_reg = torch.linalg.eigvalsh(A_reg)
            
            # Ensure numerical stability (avoid dividing by zero)
            eig_data = eig_data[eig_data > eig_tol]
            eig_reg = eig_reg[eig_reg > eig_tol]

            # Use the ratio of eigenvalues to set beta range
            lambda_d = torch.max(eig_data)
            lambda_r = torch.min(eig_reg)
        return beta0_ratio * lambda_d / lambda_r
  
    def compute_sensitivity(self,J):
        return  torch.sqrt(torch.sum(J**2, axis=0))

    def update_Ws(self, J):
        Sensitivity = self.compute_sensitivity(J)
        Sensitivity /= Sensitivity.max()
        Sensitivity = np.clip(Sensitivity, self.Ws_threshold, 1)
        self.Ws = torch.diag(Sensitivity)

    # def update_Ws(self, m):
    #     """Approximate sensitivity using Jᵀ Wᵀ W J diag only (no full Jacobian)"""
    #     m = m.detach().clone().requires_grad_(True)
    #     nparam = m.numel()
    #     sensitivity_sq = torch.zeros(nparam)

    #     for i in range(nparam):
    #         # Basis vector ei
    #         ei = torch.zeros_like(m)
    #         ei[i] = 1.0
    #         # Compute J @ ei = directional derivative w.r.t. m_i
    #         dpred, dFdm_i = self.Jvec(m, ei)
    #         wr = self.Wd @ dFdm_i  # Wd @ column of J
    #         sensitivity_sq[i] = torch.sum(wr ** 2)
    #     self.Ws= torch.sqrt(sensitivity_sq)
    #     return self.Ws

    def GradientDescent(self,mvec_init, niter, beta0, L1reg=False, print_update=True, 
        coolingFactor=2.0, coolingRate=2, s0=1.0, sfac = 0.5,update_Wsen=False, 
        stol=1e-6, gtol=1e-3, mu=1e-4,ELS=True, BLS=True ):

        self.error_prg = []
        self.data_prg = []
        self.mvec_prg = []
        
        mvec_old = mvec_init.detach().clone().requires_grad_(True)
        m_ref = mvec_old.detach()
        self.beta= beta0

        if L1reg:
            f_old, phid, phim, dpred_m = self.loss_func_L1reg(mvec_old, m_ref=m_ref)
        else:
            f_old, phid, phim, dpred_m = self.loss_func_L2reg(mvec_old, m_ref=m_ref)
        self.error_prg.append(np.array([f_old.item(), phid.item(), phim.item()]))
        self.mvec_prg.append(mvec_old.detach().numpy())
        self.data_prg.append(dpred_m.detach().numpy())
        if print_update:
            print(f'{0:3}, beta:{self.beta:.1e}, phid:{phid:.1e},    phim:{phim:.1e}, f:{f_old:.1e}')

        for i in range(niter):
            self.beta = beta0* torch.tensor(1.0 / (coolingFactor ** (i // coolingRate)))

            if L1reg:
                f_new, phid, phim, dpred_m, g, H = self.loss_func_L1reg_gH(mvec_old,update_Wsen=update_Wsen,m_ref=m_ref)
            else:
                f_new, phid, phim, dpred_m, g, H = self.loss_func_L2reg_gH(mvec_old,update_Wsen=update_Wsen,m_ref=m_ref)

            # f_old.backward()   # Compute the gradient of f_old

            # # if mvec_old.grad is not None:
            # #     mvec_old.grad.zero_()  # Zero out the gradient before computing it  
            # g = mvec_old.grad  # Get the gradient of mvec_old

           # Exact line search
            if ELS:
                if update_Wsen:
                    Jg= self.Jmat@g
                else:
                    dpred, Jg =self.Jvec(mvec_init,g)
                t = torch.sum(g**2)/torch.sum((self.Wd @ Jg )**2)
            else:
                t = 1.

            g_norm = torch.linalg.norm(g, ord=2)

            if g_norm < torch.tensor(gtol):
                print(f"Inversion complete since norm of gradient is small as: {g_norm:.3e}")
                break

            s = torch.tensor(s0)
            dm = t*g.flatten()  # Ensure dm is a 1D tensor
            mvec_new = self.project_convex_set(mvec_old - s * dm)
            if L1reg:
                f_new, phid, phim, dpred_m = self.loss_func_L1reg(mvec_new, m_ref=m_ref)
            else:
                f_new, phid, phim, dpred_m = self.loss_func_L2reg(mvec_new, m_ref=m_ref)
            directional_derivative = torch.dot(g.flatten(), -dm.flatten())
            if BLS:
                while f_new >= f_old + s*torch.tensor(mu)* directional_derivative:
                    s *= torch.tensor(sfac)
                    mvec_new = self.project_convex_set(mvec_old - s * dm)
                    if L1reg:
                        f_new, phid, phim, dpred_m = self.loss_func_L1reg(mvec_new, m_ref=m_ref)
                    else:
                        f_new, phid, phim, dpred_m = self.loss_func_L2reg(mvec_new, m_ref=m_ref) 
                    if s < torch.tensor(stol):
                        break
            mvec_old = mvec_new.detach().clone().requires_grad_(True)
            f_old = f_new
            self.error_prg.append(np.array([f_new.item(), phid.item(), phim.item()]))
            self.mvec_prg.append(mvec_new.detach().numpy())
            self.data_prg.append(dpred_m.detach().numpy())
            if print_update:
                print(f'{i + 1:3}, beta:{self.beta:.1e}, step:{s:.1e}, gradient:{g_norm:.1e}, f:{f_new:.1e}')
        return mvec_new

    def GaussNewton(self, mvec_init, niter, beta0, L1reg=False, J_prj=False, print_update=True, 
        coolingFactor=2.0, coolingRate=2, s0=1.0, sfac = 0.5,update_Wsen=False, 
        stol=1e-6, gtol=1e-3, mu=1e-4):
        self.error_prg = []
        self.data_prg = []
        self.mvec_prg = []
        
        mvec_old = mvec_init.detach().clone().requires_grad_(True)
        # mvec_old = mvec_init #.detach()
        m_ref = mvec_init.detach()
        self.beta = beta0

        if L1reg:
            f_old, phid, phim, dpred_m = self.loss_func_L1reg(mvec_old, m_ref=m_ref)
        if J_prj:
            f_old, phid, phim, dpred_m = self.loss_func_Jacobian_proj(mvec_old, m_ref=m_ref)
        else:
            f_old, phid, phim, dpred_m = self.loss_func_L2reg(mvec_old, m_ref=m_ref)
        self.error_prg.append(np.array([f_old.item(), phid.item(), phim.item()]))
        self.mvec_prg.append(mvec_old.detach().numpy())
        self.data_prg.append(dpred_m.detach().numpy())
        if print_update:
            print(f'{0:3}, beta:{self.beta:.1e}, phid:{phid:.1e},    phim:{phim:.1e}, f:{f_old:.1e}')

        for i in range(niter):
            self.beta = beta0* torch.tensor(1.0 / (coolingFactor ** (i // coolingRate)))
            # rd = self.Wd@(self.dpred(mvec_old) - self.dobs)
            # J = self.J(mvec_old)
            # if update_Wsen:
            #     self.update_Ws(self.Wd@J)                
            # g = J.T @ self.Wd.T@ rd
            # H = J.T @ self.Wd.T@ self.Wd@J
            # if m_ref is not None:
            #     g += beta * self.alphas * (self.Ws.T@self.Ws@ (mvec_old - m_ref))
            #     H += beta * self.alphas * self.Ws.T@self.Ws
            # if self.Wx is not None:
            #     g += beta * self.alphax * (
            #         self.Wx.T @ self.Wx @ mvec_old)
            #     H += beta * self.alphax * self.Wx.T @ self.Wx
            if L1reg:
                f_new, phid, phim, dpred_m, g, H = self.loss_func_L1reg_gH(mvec_old,update_Wsen=update_Wsen,m_ref=m_ref)
            elif J_prj:
                f_new, phid, phim, dpred_m, g, H = self.loss_func_Jacobian_proj_gh(mvec_old,update_Wsen=update_Wsen,m_ref=m_ref)
            else:
                f_new, phid, phim, dpred_m, g, H = self.loss_func_L2reg_gH(mvec_old,update_Wsen=update_Wsen,m_ref=m_ref)
            dm = torch.linalg.solve(H, g).flatten()  # Ensure dm is a 1D tensor
            g_norm = torch.linalg.norm(g, ord=2)

            if g_norm < torch.tensor(gtol):
                print(f"Inversion complete since norm of gradient is small as: {g_norm:.3e}")
                break

            s = torch.tensor(s0)
            mvec_new = self.project_convex_set(mvec_old - s * dm)
            if L1reg:
                f_new, phid, phim, dpred_m = self.loss_func_L1reg(mvec_new, m_ref=m_ref)
            elif J_prj:
                f_new, phid, phim, dpred_m = self.loss_func_Jacobian_proj(mvec_new, m_ref=m_ref)
            else:
                f_new, phid, phim, dpred_m = self.loss_func_L2reg(mvec_new, m_ref=m_ref)
            directional_derivative = torch.dot(g.flatten(), -dm.flatten())
            while f_new >= f_old + s*torch.tensor(mu)* directional_derivative:
                s *= torch.tensor(sfac)
                mvec_new = self.project_convex_set(mvec_old - s * dm)
                if L1reg:
                    f_new, phid, phim, dpred_m = self.loss_func_L1reg(mvec_new, m_ref=m_ref)
                elif J_prj:
                    f_new, phid, phim, dpred_m = self.loss_func_Jacobian_proj(mvec_new, m_ref=m_ref)
                else:
                    f_new, phid, phim, dpred_m = self.loss_func_L2reg(mvec_new, m_ref=m_ref) 
                if s < torch.tensor(stol):
                    break
            mvec_old = mvec_new.detach().clone().requires_grad_(True)
            f_old = f_new
            self.error_prg.append(np.array([f_new.item(), phid.item(), phim.item()]))
            self.mvec_prg.append(mvec_new.detach().numpy())
            self.data_prg.append(dpred_m.detach().numpy())
            if print_update:
                print(f'{i + 1:3}, beta:{self.beta:.1e}, step:{s:.1e}, gradient:{g_norm:.1e}, f:{f_new:.1e}')
        return mvec_new
    
def enforce_descending_x(ax):
    ax.invert_xaxis()                      # just flip direction
    ax.autoscale(enable=True, axis="x")    # keep autoscale ON for later overlays
    return ax


def enforce_negative_up(ax):
    ax.invert_yaxis()
    ax.autoscale(enable=True, axis="y")
    return ax

