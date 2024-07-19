import time
from collections import namedtuple
from fdtd import * # import the fdtd module

if __name__ == "__main__":
    
    regionx = 200e-9 # region size in x
    regiony = 200e-9 # region size in y
    regionz = 200e-9 # region size in z

    dxtarget = 5e-9 # dx (m)
    dytarget = 5e-9 # dy (m)
    dztarget = 5e-9 # dz (m)

    source = 'dipole' # 'dipole' or 'plane' wave source
    pulse = 'pulse' # 'pulse' or 'cw' wave source

    lambda0 = 0.561e-6 # center wavelength in vacuum (m)
    courantfac = 0.98 # Courant factor
    mt = 2**6 # number of iterations, must be integer power of 2
    mfft = 2**6 # number of sampling for FFT, must be inteeger power of 2
    extrapol = 4 # zero-filling factor before FFT

    msf = 3 # width for scattering field region (>=3)
    mpml = 8 # number of perfectly matched layer (PML)
    kappamax = 100.0 # parameter for CFS-CPML
    amax = 10.0 # parameter for CFS-CPML
    mpow = 3 # parameter for CFS-CPML
    
    r1 = 25e-9 # radius of inner sphere (m)
    Obj = namedtuple('Obj', ('shape', 'material', 'position', 'size'))
    objs = (
        Obj('background', 'vacuum', 0, 0),
        Obj('substrate', 'SiO2', (0, 0, r1), 0),
        Obj('sphere', 'Au', (0, 0, 0), r1)
    )

    Dipole = namedtuple('Dipole', ('pol', 'phase', 'x', 'y', 'z'))
    # phase: 'in' in-phase, 'anti' anti-phase
    dipoles = (
        Dipole('z', 'in', 0, 0, -30e-9),
    )

    # field monitors
    savenum = 32 # total number of data saving
    saveint = mt // savenum # interval for data saving
    Fmon = namedtuple('Fmon', ('ehfield', 'axis', 'position'))
    fieldmons = (
        savenum, saveint,
        Fmon('Ex', 'y', 0),
        Fmon('Ex', 'z', 0),
        Fmon('Ez', 'y', 0),
        Fmon('Hy', 'x', 0)
    )

    # epsilon monitors
    Epsmon = namedtuple('Epsmon', ('pol', 'axis', 'position'))
    epsmons = (
        Epsmon('x', 'z', 0),
        Epsmon('x', 'y', 0),
        Epsmon('z', 'z', 0)
    )

    r1 = 25.0e-9 # radius of sphere (m)
    Dtct = namedtuple('Dtct', ('pol', 'x', 'y', 'z'))
    detectors = (
        Dtct('x', 0, 0, 0),
        Dtct('x', r1 + 5.0e-9, 0, 0),
        Dtct('x', r1 + 5.0e-9, 0, 0),
        Dtct('x', r1, 0, r1),
        Dtct('x', r1, 0, r1)
    )
    
    em = Fdtd(\
        source, pulse, lambda0, courantfac, mt, mfft, extrapol, \
        regionx, regiony, regionz, dxtarget, dytarget, dztarget, \
        mpml, msf, kappamax, amax, mpow, \
        objs, fieldmons, epsmons, detectors, dipoles)
    
    start = time.time()
    em.sweep()
    print(f'Elapsed time: {time.time() - start} s')
    










