import numpy as np
import numba 
import matplotlib.pyplot as plt
import scipy.stats as scs
import multiprocessing
from multiprocessing import shared_memory
import os
from itertools import repeat
import time

SIZE = 10000
RHOS = [1]

nx = ny = 100
RMAXS = np.linspace(0.2e-9,7e-9, nx)
SEPARATIONS = np.linspace(1.5e-8,7e-8, ny)
xv, yv = np.meshgrid(RMAXS, SEPARATIONS, sparse=False, indexing='xy')

OMEGAS = np.zeros(shape=(nx,ny), dtype=np.float64)
GAMMAS = np.zeros(shape=(nx,ny), dtype=np.float64)
OMEGAONGAMMAS = np.zeros(shape=(nx,ny), dtype=np.float64)

omegashm = shared_memory.SharedMemory(create=True, size=OMEGAS.nbytes)
gammashm = shared_memory.SharedMemory(create=True, size=GAMMAS.nbytes)
omegaongammashm = shared_memory.SharedMemory(create=True, size=OMEGAONGAMMAS.nbytes)

OMEGAS = np.ndarray(OMEGAS.shape, dtype=OMEGAS.dtype, buffer=omegashm.buf)
GAMMAS = np.ndarray(GAMMAS.shape, dtype=GAMMAS.dtype, buffer=gammashm.buf)
OMEGAONGAMMAS = np.ndarray(OMEGAONGAMMAS.shape, dtype=OMEGAONGAMMAS.dtype, buffer=omegaongammashm.buf)





Sx = 1/np.sqrt(2) * np.array([[0,1,0],[1,0,1],[0,1,0]])
Sy = 1/(np.sqrt(2)*1j) * np.array([[0,1,0],[-1,0,1],[0,-1,0]])
SxSx = np.kron(Sx,Sx)
SySy = np.kron(Sy,Sy)
minus1 = [0,0,-1]
zero = [0,1,0]
one = [1,0,0]

def generate_points(rmax, separation, SIZE):
    n = 3 # or any positive integer
    points = np.random.normal(size=(SIZE, n)) 
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    rs = np.random.uniform(0,rmax**n, SIZE)
    rs = np.power(rs,1/n)
    points = rs.reshape(-1,1) * points
    points[:,0] = points[:,0]+separation
    return points

def define_interactions(rmax, rho):
    rho *= 1e27
    V = (4 * np.pi * rmax**3)/3;
    h = 6.6260688e-34;
    mu0 = 4 * np.pi * 10**(-7)
    muB = 9.274e-24
    ge = 2
    @numba.njit()
    def xx_interaction_at_point(point):
        x, y, z = point
        rsq = x**2 + y**2 + z**2
        return (ge * muB)**2 * mu0 * rho * V / (4 * np.pi) / rsq**(3/2) *(1 - 3*x**2 / rsq) / h / 1e6

    @numba.njit()
    def xy_interaction_at_point(point):
        x, y, z = point
        rsq = x**2 + y**2 + z**2
        return (ge * muB)**2 * mu0 * rho * V / (4 * np.pi) / rsq**(3/2) *(-3*x*y / rsq) / h / 1e6

    @numba.njit()
    def xz_interaction_at_point(point):
        x, y, z = point
        rsq = x**2 + y**2 + z**2
        return (ge * muB)**2 * mu0 * rho * V / (4 * np.pi) / rsq**(3/2) *(-3*x*z / rsq) / h / 1e6


    @numba.njit()
    def yy_interaction_at_point(point):
        x, y, z = point
        rsq = x**2 + y**2 + z**2
        return (ge * muB)**2 * mu0 * rho * V / (4 * np.pi) / rsq**(3/2) *(1 - 3*y**2 / rsq) / h / 1e6

    @numba.njit()
    def yz_interaction_at_point(point):
        x, y, z = point
        rsq = x**2 + y**2 + z**2
        return (ge * muB)**2 * mu0 * rho * V / (4 * np.pi) / rsq**(3/2) *(-3*y*z / rsq) / h / 1e6

    @numba.njit()
    def zz_interaction_at_point(point):
        x, y, z = point
        rsq = x**2 + y**2 + z**2
        return (ge * muB)**2 * mu0 * rho * V / (4 * np.pi) / rsq**(3/2) *(1 - 3*z*z / rsq) / h / 1e6

    return xx_interaction_at_point, xy_interaction_at_point, xz_interaction_at_point, yy_interaction_at_point, yz_interaction_at_point, zz_interaction_at_point



def compute(RHO, i, j):
    RMAX, SEPARATION = xv[i,j], yv[i,j]
    xx, _, xz, yy, yz, zz = define_interactions(RMAX, RHO)
    points = generate_points(RMAX, SEPARATION, SIZE)

    interactions_xx = np.apply_along_axis(xx, 1, points)
    interactions_xz = np.apply_along_axis(xz, 1, points)

    interactions_yy = np.apply_along_axis(yy, 1, points)
    interactions_yz = np.apply_along_axis(yz, 1, points)

    interactions_zz = np.apply_along_axis(zz, 1, points)
    Cxx, Cyy = np.mean(interactions_xx), np.mean(interactions_yy)
    
    _, scale = scs.cauchy.fit(interactions_xz + interactions_yz + interactions_zz)

    gammae = 28000 #Mhz
    deltaB = 0.5 / 1e4 * (SEPARATION*1e9) # 0.5G/nm -> T * SEPARATION nm
    centre_difference = gammae*(deltaB)
    cauchydistone = scs.cauchy(loc=0,scale=scale)
    cauchydistother = scs.cauchy(loc=centre_difference, scale=scale)
    xs = np.linspace(-20,100,10000)
    dx = (100-(-20)) / 10000
    y1s, y2s = cauchydistone.pdf(xs), cauchydistother.pdf(xs)
    densityofstates = dx*np.dot(y1s,y2s)

    V = Cxx * SxSx + Cyy*SySy
    state1s = [np.kron(minus1,minus1), np.kron(minus1,zero), np.kron(minus1,minus1),np.kron(zero,minus1), np.kron(minus1,zero), np.kron(zero,zero)]
    state2s = [np.kron(zero,zero), np.kron(zero,minus1), np.kron(one,one), np.kron(one,one), np.kron(one,one), np.kron(one,one)]
    gammabath = 1 / (2 * 1e-3) / 1e6
    print([np.abs(np.dot(state1,V @ state2))**2*densityofstates for state1, state2 in zip(state1s, state2s)])
    totalgamma = scale + np.sum([np.abs(np.dot(state1,V @ state2))**2*densityofstates for state1, state2 in zip(state1s, state2s)]) + gammabath

    omega = np.mean(interactions_zz)

    OMEGAS[i,j] = omega
    GAMMAS[i,j] = totalgamma
    OMEGAONGAMMAS[i,j] = omega / totalgamma



if __name__ == "__main__":
    try:
        with multiprocessing.Pool(os.cpu_count()-1) as pool:
            for RHO in RHOS:
                for i in range(nx):
                    start = time.time()
                    pool.starmap(compute, zip(repeat(RHO), repeat(i),range(ny)))
                    print(f"round {i} took {time.time()-start}s")
                omegas = np.ctypeslib.as_array(OMEGAS)
                gammas = np.ctypeslib.as_array(GAMMAS)
                omegaongammas = np.ctypeslib.as_array(OMEGAONGAMMAS)
                np.save(f'./data/rho{RHO:.2g}omega.npy', omegas)
                np.save(f'./data/rho{RHO:.2g}gammas.npy', gammas)
                np.save(f'./data/rho{RHO:.2g}ratios.npy', omegaongammas)
            np.save(f'./data/xs.npy', xv)
            np.save(f'./data/ys.npy', yv)
    except Exception as e:
        print(e)
        pass
    

    omegashm.close()
    gammashm.close()
    omegaongammashm.close()
    omegashm.unlink()
    gammashm.unlink()
    omegaongammashm.unlink()