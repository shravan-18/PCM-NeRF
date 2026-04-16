import torch
import torch.nn as nn
import numpy as np
from utils.lie_group_helper import make_c2w, Exp
from models.poses import LearnPose

class UncertainLearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None, init_uncertainty=0.01):
        """
        Camera pose parameters with uncertainty modeling
        
        Args:
            num_cams: Number of cameras
            learn_R: Whether to learn rotation
            learn_t: Whether to learn translation
            init_c2w: Initial camera-to-world matrices
            init_uncertainty: Initial uncertainty value for all parameters
        """
        super(UncertainLearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None

        if isinstance(init_c2w, str):
            poses = np.load(init_c2w).astype(np.float32)
            init_c2w = [torch.from_numpy(pose) for pose in poses]
            init_c2w = torch.stack(init_c2w)
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        
        # Mean pose parameters (same as LearnPose)
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)
        
        # Log uncertainty parameters (diagonal of covariance matrix in SE(3) space)
        # We use log space for numerical stability and to ensure positive variances
        log_init = np.log(init_uncertainty)
        self.log_r_uncertainty = nn.Parameter(
            torch.ones(size=(num_cams, 3), dtype=torch.float32) * log_init, 
            requires_grad=learn_R
        )  # (N, 3)
        self.log_t_uncertainty = nn.Parameter(
            torch.ones(size=(num_cams, 3), dtype=torch.float32) * log_init, 
            requires_grad=learn_t
        )  # (N, 3)

    def get_distribution(self, cam_id):
        """
        Returns the distribution parameters for a camera
        """
        r_mean = self.r[cam_id]  # (3, )
        t_mean = self.t[cam_id]  # (3, )
        r_var = torch.exp(self.log_r_uncertainty[cam_id])  # (3, )
        t_var = torch.exp(self.log_t_uncertainty[cam_id])  # (3, )
        
        return {
            'r_mean': r_mean,
            't_mean': t_mean,
            'r_var': r_var,
            't_var': t_var
        }

    def sample_poses(self, cam_id, n_samples=1):
        """
        Sample n_samples poses from the distribution
        """
        dist = self.get_distribution(cam_id)
        
        # Sample rotation parameters
        r_samples = dist['r_mean'].unsqueeze(0) + torch.sqrt(dist['r_var']).unsqueeze(0) * torch.randn(n_samples, 3, device=dist['r_mean'].device)
        
        # Sample translation parameters
        t_samples = dist['t_mean'].unsqueeze(0) + torch.sqrt(dist['t_var']).unsqueeze(0) * torch.randn(n_samples, 3, device=dist['t_mean'].device)
        
        # Convert to transformation matrices
        c2w_samples = []
        for i in range(n_samples):
            R = Exp(r_samples[i])  # (3, 3)
            c2w = torch.cat([R, t_samples[i].unsqueeze(1)], dim=1)  # (3, 4)
            
            # Apply init pose if provided
            if self.init_c2w is not None:
                c2w = make_c2w(r_samples[i], t_samples[i]) @ self.init_c2w[cam_id]
            
            c2w_samples.append(c2w)
        
        return torch.stack(c2w_samples)  # (n_samples, 4, 4) or (n_samples, 3, 4)

    def forward(self, cam_id):
        """
        Return the mean pose (for compatibility with existing code)
        """
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)

        # Apply init pose if provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w
    
    def get_uncertainty_magnitude(self, cam_id=None):
        """
        Returns the overall uncertainty magnitude for a camera or all cameras.
        Useful for diagnostics. For per-camera LR damping use
        get_per_camera_scalar_uncertainty() instead.
        """
        if cam_id is not None:
            r_uncertainty = torch.exp(self.log_r_uncertainty[cam_id])
            t_uncertainty = torch.exp(self.log_t_uncertainty[cam_id])
            return (r_uncertainty.mean() + t_uncertainty.mean()) / 2.0
        else:
            r_uncertainty = torch.exp(self.log_r_uncertainty)
            t_uncertainty = torch.exp(self.log_t_uncertainty)
            return (r_uncertainty.mean() + t_uncertainty.mean()) / 2.0

    def get_per_camera_scalar_uncertainty(self):
        """
        Returns sigma-bar_i for every camera as a (N,) tensor.

        Matches the paper's definition (Sec. 3.2):
            sigma-bar_i = (||sigma^2_{r,i}||_1 + ||sigma^2_{t,i}||_1) / 6

        The denominator 6 normalises the sum of two L1-norms over 3-D vectors
        (3 rotation + 3 translation scalar variances) so the result lies in
        the same unit range as the confidence target (1 - gamma_i) in [0, 1].

        Used by the per-camera gradient-damping rule:
            eta_i^pose = eta_0 / (1 + sigma-bar_i * kappa)
        """
        # exp of log-space params -> positive variances, shape (N, 3) each
        sigma2_r = torch.exp(self.log_r_uncertainty)   # (N, 3)
        sigma2_t = torch.exp(self.log_t_uncertainty)   # (N, 3)
        # L1-norm along the 3-component axis, sum the two norms, divide by 6
        sigma_bar = (sigma2_r.sum(dim=1) + sigma2_t.sum(dim=1)) / 6.0  # (N,)
        return sigma_bar
        