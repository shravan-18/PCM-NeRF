import torch
import torch.nn as nn

class UncertaintyAwareIoULoss(nn.Module):
    def __init__(self, sample_dist, resolution=256, topk=16, n_samples=8, uncertainty_lambda=0.05):
        """
        Uncertainty-aware IoU loss that works with pose distributions
        
        Args:
            sample_dist: Sample distance
            resolution: Resolution of the voxel grid
            topk: Number of top points to use
            n_samples: Number of Monte Carlo samples for expectation estimation
            uncertainty_lambda: Weight for the uncertainty penalty
        """
        super(UncertaintyAwareIoULoss, self).__init__()
        self.sample_dist = sample_dist
        self.resolution = resolution
        self.equals = complex(0, resolution)
        self.topk = topk
        self.sigma = 6 / self.resolution
        self.n_samples = n_samples
        self.uncertainty_lambda = uncertainty_lambda
        
        equals = torch.linspace(-1, 1, self.resolution)
        x, y, z = torch.meshgrid(equals, equals, equals)
        samples = torch.column_stack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
        self.sample_2 = samples.expand(1, topk, -1, -1)
        self.pdist = nn.PairwiseDistance(p=2)
        self.sigmoid_param = 30

    def Gaussian(self, means, weights):
        diff = self.sample_2 - means  # batchsize, topk, res^3, 3
        squared_distances = (diff ** 2).sum(dim=-1)
        gauss = torch.exp(-squared_distances / (2 * self.sigma * self.sigma))  # batchsize, topk, res^3
        gauss = (gauss * weights).sum(dim=1)  # batchsize, res^3
        return gauss

    def mixGaussian(self, pts, weights):
        heatmap = self.Gaussian(pts[:, :, None, :], weights[:, :, None])
        return heatmap

    def compute_iou(self, pts1, weights1, pts2, weights2):
        """
        Compute IoU between two point clouds with weights
        """
        idx = ~torch.isnan(weights1).any(dim=1) & ~torch.isnan(weights2).any(dim=1)
        pts1 = pts1[idx]
        pts2 = pts2[idx]
        weights1 = weights1[idx]
        weights2 = weights2[idx]

        if len(pts1) == 0:
            return 0.0, True

        if len(weights1.shape) == 1:
            pts1 = pts1.unsqueeze(dim=0)
            weights1 = weights1.unsqueeze(dim=0)
        if len(weights2.shape) == 1:
            pts2 = pts2.unsqueeze(dim=0)
            weights2 = weights2.unsqueeze(dim=0)

        self.n_batches = pts1.shape[0]
        self.n_sample = pts1.shape[1]

        # Get top-k points by weight
        top_v1, top_i1 = torch.topk(weights1, k=min(self.topk, weights1.shape[1]), dim=1, largest=True)
        expanded_i1 = top_i1.unsqueeze(dim=2).expand(-1, -1, 3)
        weights1 = top_v1
        weights1 /= (weights1.sum(dim=1)[:, None] + 1e-5)
        pts1 = torch.gather(pts1, 1, expanded_i1)

        top_v2, top_i2 = torch.topk(weights2, k=min(self.topk, weights2.shape[1]), dim=1, largest=True)
        expanded_i2 = top_i2.unsqueeze(dim=2).expand(-1, -1, 3)
        weights2 = top_v2
        weights2 /= (weights2.sum(dim=1)[:, None] + 1e-5)
        pts2 = torch.gather(pts2, 1, expanded_i2)

        # Establish bounding box for the mixture of Gaussians
        x_min = float(min(pts1[:, :, 0].min(), pts2[:, :, 0].min())) - 3*self.sigma
        y_min = float(min(pts1[:, :, 1].min(), pts2[:, :, 1].min())) - 3*self.sigma
        z_min = float(min(pts1[:, :, 2].min(), pts2[:, :, 2].min())) - 3*self.sigma
        x_max = float(max(pts1[:, :, 0].max(), pts2[:, :, 0].max())) + 3*self.sigma
        y_max = float(max(pts1[:, :, 1].max(), pts2[:, :, 1].max())) + 3*self.sigma
        z_max = float(max(pts1[:, :, 2].max(), pts2[:, :, 2].max())) + 3*self.sigma
        
        # Create the grid
        equal_x = torch.linspace(x_min, x_max, self.resolution)
        equal_y = torch.linspace(y_min, y_max, self.resolution)
        equal_z = torch.linspace(z_min, z_max, self.resolution)
        x, y, z = torch.meshgrid(equal_x, equal_y, equal_z)
        samples = torch.column_stack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
        self.sample_2 = samples.expand(1, min(self.topk, weights1.shape[1]), -1, -1)

        # Generate the mixture of Gaussians
        heatmap1 = self.mixGaussian(pts1, weights1)
        heatmap2 = self.mixGaussian(pts2, weights2)

        # Compute IoU
        I = (heatmap1 * heatmap2).sum()
        U = (heatmap1 + heatmap2 - heatmap1 * heatmap2).sum() + 1e-5
        iou = I / U

        return 1 - iou, False

    def forward(self, pts1, weights1, pts2, weights2, r1_var=None, t1_var=None, r2_var=None, t2_var=None):
        """
        Forward pass with support for uncertainty
        
        Args:
            pts1, pts2: Core points from the two rays
            weights1, weights2: Weights for the points
            r1_var, t1_var, r2_var, t2_var: Uncertainties for camera poses
        """
        if r1_var is None or t1_var is None or r2_var is None or t2_var is None:
            # Fall back to standard IoU loss if no uncertainty provided
            iou_loss, is_empty = self.compute_iou(pts1, weights1, pts2, weights2)
            return iou_loss if not is_empty else -1
        
        # Compute Monte Carlo estimate of expected IoU
        iou_samples = []
        for _ in range(self.n_samples):
            # Apply random perturbations based on uncertainties
            # This is a simplified approximation - more accurate would be to sample
            # actual camera poses and recompute the points
            
            # For pose 1
            r1_noise = torch.sqrt(r1_var.unsqueeze(0)) * torch.randn(pts1.shape[0], 3, device=pts1.device)
            t1_noise = torch.sqrt(t1_var.unsqueeze(0)) * torch.randn(pts1.shape[0], 3, device=pts1.device)
            
            # For pose 2
            r2_noise = torch.sqrt(r2_var.unsqueeze(0)) * torch.randn(pts2.shape[0], 3, device=pts2.device)
            t2_noise = torch.sqrt(t2_var.unsqueeze(0)) * torch.randn(pts2.shape[0], 3, device=pts2.device)
            
            # Apply perturbations to points (simplified model)
            # In practice, we'd recompute ray intersections but this is computationally expensive
            # This is an approximation - in a full implementation we'd properly transform points
            pts1_perturbed = pts1 + r1_noise.unsqueeze(1) * 0.01 + t1_noise.unsqueeze(1)
            pts2_perturbed = pts2 + r2_noise.unsqueeze(1) * 0.01 + t2_noise.unsqueeze(1)
            
            # Compute IoU with perturbed points
            iou_loss, is_empty = self.compute_iou(pts1_perturbed, weights1, pts2_perturbed, weights2)
            if not is_empty:
                iou_samples.append(iou_loss)
        
        if len(iou_samples) == 0:
            return -1
        
        # Compute expected IoU and uncertainty penalty
        iou_tensor = torch.stack(iou_samples)
        expected_iou = iou_tensor.mean()
        uncertainty_penalty = iou_tensor.std()
        
        # Final loss with uncertainty penalty
        final_loss = expected_iou + self.uncertainty_lambda * uncertainty_penalty
        
        return final_loss
    