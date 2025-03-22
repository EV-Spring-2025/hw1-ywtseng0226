import torch


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = torch.tensor(
    [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ]
)
SH_C3 = torch.tensor([
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
    0.3731763325901154, -0.4570457994644658, 1.445305721320277,
    -0.5900435899266435
])
SH_C4 = torch.tensor([
    2.5033429417967046, -1.7701307697799304, 0.9461746957575601,
    -0.6690465435572892, 0.10578554691520431, -0.6690465435572892,
    0.47308734787878004, -1.7701307697799304, 0.6258357354491761,
])


def RGB_to_SH(rgb: torch.Tensor) -> torch.Tensor:
    """ Converts RGB color to SH (Spherical Harmonics) representation. """
    return (rgb - 0.5) / SH_C0


def SH_to_RGB(sh: torch.Tensor) -> torch.Tensor:
    """ Converts SH representation back to RGB color. """
    return sh * SH_C0 + 0.5


def eval_sh(deg: int, sh_coeffs: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """
    Evaluate SH function at given unit directions.
    
    Args:
        deg: SH degree (0-4 supported)
        sh_coeffs: SH coefficients [..., C, (deg + 1)^2]
        directions: unit directions [..., 3]

    Returns:
        Evaluated color at each direction [..., C]
    """
    assert 0 <= deg <= 4, "Only SH degrees 0-4 are supported"
    coeff_count = (deg + 1) ** 2
    assert sh_coeffs.shape[-1] >= coeff_count, "Insufficient SH coefficients for given degree"

    # Base SH term (l=0)
    result = SH_C0 * sh_coeffs[..., 0]

    if deg >= 1:
        x, y, z = directions[..., 0:1], directions[..., 1:2], directions[..., 2:3]
        result += -SH_C1 * (y * sh_coeffs[..., 1] - z * sh_coeffs[..., 2] + x * sh_coeffs[..., 3])

    if deg >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result += (
            SH_C2[0] * xy * sh_coeffs[..., 4] +
            SH_C2[1] * yz * sh_coeffs[..., 5] +
            SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[..., 6] +
            SH_C2[3] * xz * sh_coeffs[..., 7] +
            SH_C2[4] * (xx - yy) * sh_coeffs[..., 8]
        )

    if deg >= 3:
        result += (
            SH_C3[0] * y * (3 * xx - yy) * sh_coeffs[..., 9] +
            SH_C3[1] * xy * z * sh_coeffs[..., 10] +
            SH_C3[2] * y * (4 * zz - xx - yy) * sh_coeffs[..., 11] +
            SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeffs[..., 12] +
            SH_C3[4] * x * (4 * zz - xx - yy) * sh_coeffs[..., 13] +
            SH_C3[5] * z * (xx - yy) * sh_coeffs[..., 14] +
            SH_C3[6] * x * (xx - 3 * yy) * sh_coeffs[..., 15]
        )

    if deg >= 4:
        result += (
            SH_C4[0] * xy * (xx - yy) * sh_coeffs[..., 16] +
            SH_C4[1] * yz * (3 * xx - yy) * sh_coeffs[..., 17] +
            SH_C4[2] * xy * (7 * zz - 1) * sh_coeffs[..., 18] +
            SH_C4[3] * yz * (7 * zz - 3) * sh_coeffs[..., 19] +
            SH_C4[4] * (zz * (35 * zz - 30) + 3) * sh_coeffs[..., 20] +
            SH_C4[5] * xz * (7 * zz - 3) * sh_coeffs[..., 21] +
            SH_C4[6] * (xx - yy) * (7 * zz - 1) * sh_coeffs[..., 22] +
            SH_C4[7] * xz * (xx - 3 * yy) * sh_coeffs[..., 23] +
            SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh_coeffs[..., 24]
        )

    return result
