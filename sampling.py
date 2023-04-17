import numpy as np
import torch
import constants as c
from matplotlib import pyplot as plt
import math
from render_utils import set_axes_equal, _set_axes_radius
import pdb

class SceneSampler:

    def __init__(self, num_dense_rays, num_sparse_rays, max_distance, device, beamwidth=None):
        self.num_dense_rays = num_dense_rays
        self.num_sparse_rays = num_sparse_rays
        self.max_distance = max_distance
        self.beamwidth = beamwidth
        self.voxel_size = None
        self.max_radius = None
        self.cache_sparse_vectors = None
        self.cache_dense_vectors = None
        self.device = device

        # Not doing importance sampling and the beamwidth is fixed so we can just cache these
        if num_sparse_rays is None and beamwidth is not None:
            # Beampattern radius
            self.max_radius = self.max_distance * math.tan(torch.deg2rad(beamwidth / 2))
            #self.max_radius = math.sqrt(self.max_radius ** 2 + self.max_radius ** 2)

            assert int(math.sqrt(num_dense_rays)) ** 2 == num_dense_rays

            steps = math.ceil(math.sqrt(4 / np.pi) * math.sqrt(num_dense_rays))

            self.voxel_size = (2 * self.max_radius) / steps

            x = torch.linspace(-self.max_radius, self.max_radius, steps=steps)
            y = torch.linspace(-self.max_radius, self.max_radius, steps=steps)
            focal_plane = torch.stack(torch.meshgrid((x, y))).permute(1, 2, 0)
            focal_plane = torch.cat((focal_plane, torch.ones(focal_plane.shape[0],
                                                             focal_plane.shape[1], 1) * self.max_distance), dim=-1)
            vectors = focal_plane.reshape(-1, 3)

            indeces = torch.where(vectors[..., 0] ** 2 + vectors[..., 1] ** 2 <= self.max_radius ** 2)[0].long()

            self.cache_dense_vectors = vectors[indeces].to(self.device)

        # Beamwidth is fixed and doing importance sampling, so just cache the sparse rays
        if self.beamwidth is not None and num_sparse_rays is not None:
            # Beampattern radius
            self.max_radius = self.max_distance * math.tan(torch.deg2rad(beamwidth / 2))
            #self.max_radius = math.sqrt(self.max_radius ** 2 + self.max_radius ** 2)

            assert int(math.sqrt(num_sparse_rays)) ** 2 == num_sparse_rays

            steps = math.ceil(math.sqrt(4 / np.pi) * math.sqrt(num_sparse_rays))

            self.voxel_size = (2 * self.max_radius) / steps

            x = torch.linspace(-self.max_radius, self.max_radius, steps=steps)
            y = torch.linspace(-self.max_radius, self.max_radius, steps=steps)
            focal_plane = torch.stack(torch.meshgrid((x, y))).permute(1, 2, 0)
            focal_plane = torch.cat((focal_plane, torch.ones(focal_plane.shape[0],
                                                             focal_plane.shape[1], 1) * self.max_distance), dim=-1)
            vectors = focal_plane.reshape(-1, 3)

            indeces = torch.where(vectors[..., 0] ** 2 + vectors[..., 1] ** 2 <= self.max_radius ** 2)[0].long()

            self.cache_sparse_vectors = vectors[indeces].to(self.device)

    def generate_unit_vectors_within_cone(self,
                                          num_vectors,
                                          beamwidth,
                                          debug_dir=None,
                                          distribution=None,
                                          cache_vectors=False):

        if distribution is not None:
            assert distribution.ndim == 1

            assert self.cache_sparse_vectors is not None, "Need to run ellipsoidal_sampling with cache_vectors=True" \
                                                          "before calling with a distribution input argument" \
                                                   "if not using a fixed tx_bw"
            assert self.voxel_size is not None

            indeces = torch.multinomial(distribution.reshape(-1), num_vectors, replacement=True).squeeze()

            vectors = self.cache_sparse_vectors[indeces, :]
            vectors[:, 0:2] = vectors[:, 0:2] + self.voxel_size * (2 * torch.rand_like(vectors[:, 0:2]) - 1)

            vectors = normalize_vectors(vectors)

            if debug_dir is not None:
                pass
                #plt.figure()
                #plt.scatter(vectors[..., 0].detach().cpu().numpy(), vectors[..., 1].detach().cpu().numpy(),
                #            s=.1,
                #            label='Dense distribution')
                #plt.scatter(self.cache_sparse_vectors[..., 0].detach().cpu().numpy(),
                #            self.cache_sparse_vectors[..., 1].detach().cpu().numpy(),
                #            c=distribution.reshape(-1).detach().cpu().numpy(), s=.1, label='Sparse distribution')
                #plt.xlim([-.4, .4 ])
                #plt.ylim([-.4, .4 ])
                #plt.legend()
                #plt.savefig(debug_dir)

            return vectors

        # SPARSE SAMPLING or CONVENTIONAL DENSE SAMPLING #

        # We can just resuse sparse cache vectors for fixed beamwidth
        if cache_vectors and self.beamwidth is not None:
            return self.cache_sparse_vectors

        # Beampattern radius
        self.max_radius = self.max_distance * math.tan(torch.deg2rad(beamwidth / 2))
        #self.max_radius = math.sqrt(self.max_radius  ** 2 + self.max_radius  ** 2)

        assert int(math.sqrt(num_vectors)) ** 2 == num_vectors

        # Upsample steps to account for croppping to circle
        steps = math.ceil(math.sqrt(4 / np.pi) * math.sqrt(num_vectors))

        self.voxel_size = (2 * self.max_radius) / steps

        x = torch.linspace(-self.max_radius, self.max_radius , steps=steps)
        y = torch.linspace(-self.max_radius, self.max_radius , steps=steps)
        focal_plane = torch.stack(torch.meshgrid((x, y))).permute(1, 2, 0)
        focal_plane = torch.cat((focal_plane, torch.ones(focal_plane.shape[0],
                                                         focal_plane.shape[1], 1) * self.max_distance), dim=-1)
        vectors = focal_plane.reshape(-1, 3)

        indeces = torch.where(vectors[..., 0] ** 2 + vectors[..., 1] ** 2 <= self.max_radius** 2)[0].long()

        vectors = normalize_vectors(vectors[indeces])

        if cache_vectors:
            self.cache_sparse_vectors = vectors

        return vectors


    def ellipsoidal_sampling(self, radii, tx_pos, rx_pos, num_rays, scene_bounds,
                             tx_vec=None, create_return_vec=False, point_at_center=False,
                             cache_vectors=False,
                             debug_index=None, distribution=None, transmit_from_tx=False, debug_dir=None,
                             scene_model=None,
                             transmission_model=None,
                             occlusion_scale=None,
                             compute_normals=True,
                             scene_scale_factor=1,
                             device='cpu'):

        """

        :param radii:
        :param tx_pos:
        :param rx_pos:
        :param scene_bounds:
        :param num_vectors:
        :param tx_bw:
        :param tx_vec:
        :param create_return_vec:
        :param debug:
        :param index:
        :param device:
        :return:
        """

        """
        1. We project rays and do ellipsoid intersections at with TX and RX sitting on the x-axis. Thus, first step is to 
        rotate the tx direction vector (which is in world frame) to origin frame. 
        2. Project rays from origin and rotate rays to the tx direction vector that we rotated relative to origin
        3. Find the intersection of rays with ellipsoid to radii in the object (there and back)
        4. Rotate sampled points back to world frame using opposite rotation matrix from step 1.
        5. Snap points to PCA world coordainate.
        """
        EPS = 1e-7
        tx_pos = tx_pos.squeeze()
        rx_pos = rx_pos.squeeze()
        if not torch.is_tensor(tx_pos):
            tx_pos = torch.from_numpy(tx_pos).to(device).float()
        if not torch.is_tensor(rx_pos):
            rx_pos = torch.from_numpy(rx_pos).to(device).float()
        if not torch.is_tensor(radii):
            radii = torch.from_numpy(radii).to(device).float()
        if not torch.is_tensor(scene_bounds):
            scene_bounds = torch.from_numpy(scene_bounds).to(device).float()
        if tx_vec is not None and not torch.is_tensor(tx_vec):
            tx_vec = torch.from_numpy(tx_vec).to(device).float().squeeze()

        tx_pos = tx_pos.float()
        rx_pos = rx_pos.float()
        radii = radii.float()
        scene_bounds = scene_bounds.float()

        if tx_vec is not None:
            tx_vec = tx_vec.squeeze().float()

        # phace center of tx and rx
        pca = (tx_pos + rx_pos) / 2
        # tx moved to the origin
        tx_norm = normalize_vectors(tx_pos - pca)

        # distnace between tx and rx
        d = torch.sqrt(torch.sum((tx_pos - rx_pos) ** 2))

        # They will be at the origin initially.
        tx_origin = torch.tensor([d / 2, 0., 0.]).float().to(device)
        rx_origin = torch.tensor([-d / 2, 0., 0.]).float().to(device)

        if tx_vec is None:
            scene_center = torch.mean(scene_bounds, dim=0)
            tx_vec = scene_center - tx_pos
            # Just point torwards scene center and straight ahead (airsas setup)
            if not point_at_center:
                tx_vec[..., 2] = tx_pos[..., 2]
            tx_vec = normalize_vectors(tx_vec)
        else:
            tx_vec = normalize_vectors(tx_vec)

        # Rotation matrix TX to origin
        tx_to_origin = rotation_vec_a_to_vec_b(tx_norm, normalize_vectors(tx_origin))

        if self.beamwidth is None:
            boundary_vecs = normalize_vectors(scene_bounds - tx_pos).float()
            angles = torch.arccos(torch.sum(tx_vec[None, :] * boundary_vecs,
                                            dim=-1).clamp(min=(-1 + EPS), max=(1 - EPS)))
            max_angle = angles.abs().max()
            tx_bw = 2*torch.rad2deg(max_angle)
        else:
            tx_bw = self.beamwidth

        # Rotate tx_vec to origin frame
        tx_vec = tx_to_origin @ tx_vec.squeeze()

        # Define the ellipsoid at the origin
        a = radii / 2
        b = torch.sqrt(a ** 2 - (d ** 2) / 4)
        c = torch.sqrt(a ** 2 - (d ** 2) / 4)
        #pdb.set_trace()

        # Generate ray direcionts
        # If not using importance sampling then just load the cached dense vectors
        # cache dense vectors only set if num_importance_sample vecs is None
        if distribution is None and self.cache_dense_vectors is not None:
            dir_to = self.cache_dense_vectors
        else:
            dir_to = self.generate_unit_vectors_within_cone(
                                                       num_vectors=num_rays,
                                                       beamwidth=tx_bw,
                                                       cache_vectors=cache_vectors,
                                                       distribution=distribution,
                                                       debug_dir=debug_dir).to(device)
            #pdb.set_trace()
        num_rays = dir_to.shape[0]
        # rotate vectors to align with tx direction vector (which is in origin frame)
        rot_vecs_to_tx_vec = rotation_vec_a_to_vec_b(torch.tensor([0., 0., 1.]).to(device), tx_vec)
        dir_to = torch.t(rot_vecs_to_tx_vec @ torch.t(dir_to))

        # Compute distance from rays to ellipse intersections. Using positive solution from quadratic formula
        if transmit_from_tx:
            dist_to, _ = ray_ellipse_intersection(a=a[:, None], b=b[:, None], c=c[:, None], vec_o=tx_origin,
                                                  vec_dir=dir_to)
            vec_to = tx_origin[None, None, :] + dist_to[..., None]*dir_to[None, ...]
        else:
            dist_to, _ = ray_ellipse_intersection(a=a[:, None], b=b[:, None], c=c[:, None],
                                                  vec_o=torch.tensor([0., 0., 0.]).to(a.device),
                                                  vec_dir=dir_to)
            vec_to = torch.tensor([0., 0., 0.])[None, None, :].to(dist_to.device) + \
                     dist_to[..., None] * dir_to[None, ...]

        # Rotate vectors to world coordinates
        origin_to_tx = rotation_vec_a_to_vec_b(normalize_vectors(tx_origin), tx_norm)
        vec_to_world = torch.reshape(vec_to, (-1, 3)).float()
        vec_to_world = torch.reshape(torch.t(origin_to_tx @ torch.t(vec_to_world)),
                                     (radii.squeeze().shape[0], num_rays, 3))
        vec_to_world = vec_to_world + pca
        dir_to_world = torch.t(origin_to_tx @ torch.t(dir_to))

        model_out_info = {}

        if create_return_vec:
            # Find the intersection of vectors back to the receiver with the sampled radii
            # We compute the expected depth of the transmission vectors and then project the rays from the expected
            # depth back torwards the receiver.
            num_rad, num_ray, _ = vec_to_world.shape
            model_out = scene_model(vec_to_world.reshape(-1, 3)*scene_scale_factor, compute_normals=compute_normals)

            scatterers_to = model_out['scatterers_to']
            normals = model_out['normals']

            scatterers_to = scatterers_to.reshape(num_rad, num_ray)

            transmission_probs = transmission_model(radii=radii,
                                                    scatterers_to=scatterers_to.reshape(num_rad, num_ray),
                                                    occlusion_scale=occlusion_scale)

            depth_probs = (scatterers_to.reshape(num_rad, num_ray) * transmission_probs).abs()
            depth_probs = depth_probs / torch.sum(depth_probs, dim=0, keepdim=True)
            # Sort of like expected depth
            # expected_depth_index = torch.argmax(
            #    depth_probs, dim=0)
            expected_depth_index = \
                torch.round(torch.sum(torch.arange(0, num_rad, 1).to(depth_probs.device)[:, None] *
                           depth_probs, dim=0)).long().squeeze()

            # Gather coordinates corresponding to expected depth index
            # Select from row (num_radius) using value specified in each column
            # TODO 3D gather?
            vec_back_orig_x = torch.gather(vec_to[..., 0], 0, expected_depth_index[None, :]).squeeze()
            vec_back_orig_y = torch.gather(vec_to[..., 1], 0, expected_depth_index[None, :]).squeeze()
            vec_back_orig_z = torch.gather(vec_to[..., 2], 0, expected_depth_index[None, :]).squeeze()
            vec_back_orig = torch.cat((vec_back_orig_x[:, None],
                                             vec_back_orig_y[:, None],
                                             vec_back_orig_z[:, None]), dim=-1)


            dir_back = (rx_origin[None, :] - vec_back_orig)
            dir_back = normalize_vectors(dir_back)

            _, dist_back, disc = ray_ellipse_intersection(
                                                    a=a[:, None],
                                                    b=b[:, None],
                                                    c=c[:, None],
                                                    vec_o=vec_back_orig,
                                                    vec_dir=dir_back,
                                                    return_disc=True)
            # Only find return paths from the last radius up to the first
            vec_back = vec_back_orig + dist_back[..., None] * dir_back
            vec_back_world = torch.reshape(vec_back, (-1, 3)).float()

            vec_back_world = torch.reshape(torch.t(origin_to_tx @ torch.t(vec_back_world)),
                                     (num_rad, num_rays, 3))
            vec_back_world = vec_back_world + pca

            scatterers_back = scene_model.model_out(vec_back_world.reshape(-1, 3)*scene_scale_factor)

            ignore_return_scatterers_index = (dist_back.reshape(-1) < 0)
            # Set scatterers behind the ray to zero since thy do not affect transmission prob
            scatterers_back[ignore_return_scatterers_index] = \
                scatterers_back[ignore_return_scatterers_index] * 0.

            scatterers_back = scatterers_back.reshape(num_rad, num_ray)

            back_trans_probs = transmission_model(radii=radii,
                                                    scatterers_to=scatterers_back,
                                                    occlusion_scale=occlusion_scale)

            transmission_probs = transmission_probs * back_trans_probs

            model_out_info = {
                'scatterers_to': scatterers_to,
                'transmission_probs': transmission_probs,
                'normals': normals
            }

        if debug_dir is not None:
            """If you want to plot the ellipse"""
            # thetas = torch.linspace(0, np.pi, steps=25).squeeze()
            # phis = torch.linspace(0, 2*np.pi, steps=25).squeeze()
            # meshes = torch.permute(torch.stack(torch.meshgrid(thetas, phis, indexing='ij')), (1, 2, 0))
            # x = a[:, None, None] * torch.sin(meshes[..., 0])[None, ...] * torch.cos(meshes[..., 1])[None, ...]
            # y = b[:, None, None] * torch.sin(meshes[..., 1])[None, ...] * torch.sin(meshes[..., 0])[None, ...]
            # z = c[:, None, None] * torch.cos(meshes[..., 0])[None, ...]
            #np.save(debug_dir, vec_to.detach().cpu().numpy())


            #fig = plt.figure()
            #ax = fig.add_subplot(projection='3d')
            #ax.scatter(tx_pos[0].detach().cpu().numpy(),
            #           tx_pos[1].detach().cpu().numpy(),
            #           tx_pos[2].detach().cpu().numpy(), label='TX')
            #ax.scatter(rx_pos[0].detach().cpu().numpy(),
            #           rx_pos[1].detach().cpu().numpy(),
            #           rx_pos[2].detach().cpu().numpy(), label='RX')
            #ax.scatter(vec_to[..., 0].detach().cpu().numpy(),
            #           vec_to[..., 1].detach().cpu().numpy(),
            #           vec_to[..., 2].detach().cpu().numpy(), label='Vec To', alpha=0.2)

            #if vec_back is not None:
            #    ax.scatter(vec_back[..., 0].detach().cpu().numpy(),
            #               vec_back[..., 1].detach().cpu().numpy(),
            #               vec_back[..., 2].detach().cpu().numpy(), label='Vec Back', alpha=0.2)

            #ax.scatter(scene_bounds[:, 0].detach().cpu().numpy(),
            #           scene_bounds[:, 1].detach().cpu().numpy(),
            #           scene_bounds[:, 2].detach().cpu().numpy())

            #plt.legend()
            #ax.set_box_aspect([1, 1, 1])
            #ax.set_proj_type('ortho')
            #set_axes_equal(ax)
            #plt.savefig(debug_dir)

        return vec_to_world, dir_to_world, model_out_info

def normalize_vectors(vec):
    return vec / torch.sqrt(torch.sum(vec**2, dim=-1))[..., None]


# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
# rotation matrix to rotate vector a onto vector b
def rotation_vec_a_to_vec_b(vec_a, vec_b):
    vec_a = vec_a.squeeze()
    vec_b = vec_b.squeeze()

    assert vec_a.ndim == 1
    assert vec_b.ndim == 1
    #print("vec_a", vec_a)
    #print("vec_b", vec_b)

    v = torch.cross(vec_a, vec_b)
    #print("v", v)
    c = torch.dot(vec_a, vec_b)
    #print("c", c)
    v_x = skew_symm_cp(v)
    #print("v_x", v_x)

    # handle anti-parallel vector case
    if (1 + c).item() < 1e-2:
        #print("SATISFIED")
        return -torch.eye(3)

    return torch.eye(3).to(vec_a.device) + v_x + (v_x @ v_x) * (1 / (1 + c))



"""
Samples from an ellipse aligned with major axes, rotates ellipse to its axes, and then snaps to position. 
"""
def ray_ellipse_intersection(a, b, c, vec_o, vec_dir, return_disc=False):
    if vec_o.ndim == 1:
        vec_o = vec_o[None, ...]

    alpha = \
        (vec_dir[..., 0]**2)/(a**2) + \
        (vec_dir[..., 1]**2)/(b**2) + \
        (vec_dir[..., 2]**2)/(c**2)

    beta = \
        (2*vec_o[..., 0]*vec_dir[..., 0])/(a**2) + \
        (2*vec_o[..., 1]*vec_dir[..., 1])/(b**2) + \
        (2*vec_o[..., 2]*vec_dir[..., 2])/(c**2)

    kappa = \
        (vec_o[..., 0]**2)/(a**2) + \
        (vec_o[..., 1]**2)/(b**2) + \
        (vec_o[..., 2]**2)/(c**2) - 1.

    disc = beta**2 - 4*alpha*kappa

    #assert (disc >= 0).all()

    t_pos = (-beta + torch.sqrt(disc))/(2*alpha)
    t_neg = (-beta - torch.sqrt(disc))/(2*alpha)

    if return_disc:
        return t_pos, t_neg, disc
    else:
        return t_pos, t_neg


def skew_symm_cp(x):
    assert len(x) == 3
    assert x.ndim == 1

    if not torch.is_tensor(x):
        return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
    else:
        return torch.tensor([[0., -x[2], x[1]],
                        [x[2], 0., -x[0]],
                        [-x[1], x[0], 0.]]).to(x.device).float()


def find_voxels_within_fov(trans_pos, tx_vec, origin, voxels, bw, device='cpu'):
    assert trans_pos.ndim == 1

    orig_numpy = False

    if not torch.is_tensor(trans_pos):
        orig_numpy = True
        trans_pos = torch.from_numpy(trans_pos)

    if not torch.is_tensor(tx_vec):
        tx_vec = torch.from_numpy(tx_vec)

    if not torch.is_tensor(origin):
        origin = torch.from_numpy(origin)

    if not torch.is_tensor(voxels):
        voxels = torch.from_numpy(voxels)

    trans_pos = trans_pos.to(device).float()
    tx_vec = tx_vec.to(device).float()
    origin = origin.to(device).float()
    voxels = voxels.to(device).float()

    # rotate the voxels to the origin
    rot_mat = rotation_vec_a_to_vec_b(tx_vec, origin.type(tx_vec.dtype))
    shift_voxels = voxels - trans_pos[None, ...]
    origin_voxels = (rot_mat @ shift_voxels.T).T

    # now check if origin voxels fall within the cone given by the transducer beamwdith.
    in_fov_index = torch.where(torch.sqrt(origin_voxels[..., 0] ** 2 + origin_voxels[..., 1] ** 2) <=
                               torch.abs(origin_voxels[..., 2]) * np.tan(bw/2))

    in_fov_voxels = voxels[in_fov_index]

    if orig_numpy:
        in_fov_voxels = in_fov_voxels.detach().cpu().numpy()

    return in_fov_index, in_fov_voxels
