import numpy as np
from calc_sigmas import calc_sigmas 

#modified from https://personalpages.manchester.ac.uk/staff/paul.connolly/teaching/practicals/gaussian_plume_modelling.html

def gauss_func(Q, u, dir1, x, y, z, xs, ys, H, Dy, Dz, STABILITY):
    u1 = u
    #shift coordinates so stack is the origin
    x1 = x - xs
    y1 = y - ys

    wx = u1 * np.sin((dir1 - 180.) * np.pi / 180.)
    wy = u1 * np.cos((dir1 - 180.) * np.pi / 180.)

    dot_product = wx * x1 + wy * y1
    magnitude = u1 * np.sqrt(x1**2 + y1**2)
    subtended = np.arccos(dot_product / (magnitude + 1e-15))
    hypotenuse = np.sqrt(x1**2 + y1**2)

    downwind = np.cos(subtended) * hypotenuse
    crosswind = np.sin(subtended) * hypotenuse

    sig_y, sig_z = calc_sigmas(STABILITY, downwind)

    # Avoid division by zero
    sig_y = np.clip(sig_y, 1e-6, None)
    sig_z = np.clip(sig_z, 1e-6, None)
    sig_y *= Dy
    sig_z *= Dz


    C_out = np.zeros((x.shape[0], x.shape[1], len(z)))

    for k in range(len(z)):
        Cz = Q / (2. * np.pi * u1 * sig_y * sig_z)
        Cz *= np.exp(-crosswind**2 / (2. * sig_y**2))
        Cz *= (np.exp(-(z[k] - H)**2 / (2. * sig_z**2)) + np.exp(-(z[k] + H)**2 / (2. * sig_z**2)))
        Cz[downwind <= 0] = 0
        C_out[:, :, k] = Cz

    return C_out
