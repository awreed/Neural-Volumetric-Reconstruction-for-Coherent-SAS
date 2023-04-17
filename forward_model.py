import torch
from sas_utils import safe_normalize

def cumprod_exclusive(
        tensor: torch.Tensor
) -> torch.Tensor:
    r"""
    (Courtesy of https://github.com/krrish94/nerf-pytorch)
    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.

    return cumprod

def scattering_model(scatterers, normals, dir_vec, ratio=0.):
    assert dir_vec.shape[0] == normals.shape[1]
    # Normals is shape [num_r, num_vec, 3]
    # dir_vec is shape [num_vec, 3]

    lambertian = ratio + (1 - ratio) * torch.sum((normals*dir_vec[None, ...]), dim=-1).clamp(min=0)

    scatterers_lamb = scatterers * lambertian

    return scatterers_lamb

def transmission_model(radii, scatterers_to, occlusion_scale, factor2=False):
    radii_to = radii[1:] - radii[:-1]
    radii_to = torch.cat((radii_to, torch.tensor([torch.mean(radii_to)]).to(radii.device)), dim=-1)
    # normalize scatterer magnitude within 0 and 1

    # If we are assuming tranmission ray is the same as receive ray
    if factor2:
        occlusion_scale = occlusion_scale * 2

    alpha = torch.exp(-(scatterers_to.abs() * radii_to[:, None] * occlusion_scale))
    trans_prob = cumprod_exclusive((alpha + 1e-10).permute(1, 0)).permute(1, 0)

    return trans_prob
