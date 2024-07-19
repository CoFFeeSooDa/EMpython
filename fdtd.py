import sys
import math 
import os
import numpy as np
from preprocess import *

class Fdtd(Preprocess):
    # methods
    def sweep(self):
        """ Time development with CFS-PML and ADE """

        self.save_idv()
        numt = 0
        for jt in range(self.mt):
            # update E-field
            self.sweep_isolate_e()
            self.sweep_boundary_e()
            # E-field source injection
            if self.source == 'plane':
                self.normalinc_p_e(jt)
            else:
                self.dipole_source(jt)

            # auxilary E-field update
            self.develop_pcurrent()

            # update H-field
            self.sweep_isolate_h()
            self.sweep_boundary_h()
            # H-field source injection
            if self.source == 'plane':
                self.normalinc_p_h(jt)

            # store H and E fields
            if ((jt + 1) % self.saveint == 0) and (numt < self.savenum):
                self.save_ehfield(numt)
                numt += 1
            
            self.detect_efield(jt)

            # update arrays
            self.update_field()

        # calculate spectra
        self.detect_spectra()

    def dipole_source(self, jt):
        """ Dipole source """
        env_factor = 1.0 / 4.0
        tau = math.pi / self.omega0
        if self.pulse == 'pulse':
            t0 = 5.0 * tau
        else:
            t0 = 0.0
            omega_env = self.omega0 * env_factor
        tempe = (jt - 1)*self.dt - t0
        if self.pulse == 'pulse':
            campe = math.sin(self.omega0 * tempe)
            j00 = math.exp(-tempe**2/tau**2) * campe
        else:
            tempe2 = tempe - math.pi / omega_env
            campe = math.cos(self.omega0 * tempe2)
            if tempe2 < -math.pi / omega_env:
                j00 = 0.0
            elif tempe2 < 0:
                j00 = 0.5 * (1 + math.cos(omega_env*tempe2)) * campe
            else:
                j00 = campe
        
        for dipole in self.idipoles:
            if dipole.pol == 'x':
                self.Ex2[dipole.iz, dipole.iy, dipole.ix] =\
                    self.Ex2[dipole.iz, dipole.iy, dipole.ix] - dipole.phase*j00
            elif dipole.pol == 'y':
                self.Ey2[dipole.iz, dipole.iy, dipole.ix] =\
                    self.Ey2[dipole.iz, dipole.iy, dipole.ix] - dipole.phase*j00
            elif dipole.pol == 'z':
                self.Ez2[dipole.iz, dipole.iy, dipole.ix] =\
                    self.Ez2[dipole.iz, dipole.iy, dipole.ix] - dipole.phase*j00
            else:
                print('Error at dipole_source!')

        self.esource[jt] = j00
                
    def normalinc_p_e(self, jt):
        """ Source: x-polarized and z-propagating plane wave
            TF/SF compensation for E"""
        # generation of the temporal shape of the source wave
        iz00 = self.mz1 # origin for incident wave
        env_factor = 1.0 / 4.0
        tau = math.pi / self.omega0
        if self.pulse == 'pulse':
            t0 = 5.0 * tau
        else:
            t0 = 0.0
            omega_env = self.omega0 * env_factor

        tempe = (jt + 0.5)*self.dt - t0\
            - (self.mz1 - iz00) * self.dz / math.sqrt(self.epsr[self.bgmater])/self.cc
        temph = jt*self.dt - t0 - (self.izst - iz00 - 0.5) \
            * self.dz * math.sqrt(self.epsr[self.bgmater])/self.cc
        campe = math.sin(self.omega0*tempe)
        camph = math.sin(self.omega0*temph)
        if self.pulse == 'pulse':
            SEx00 = math.exp(-tempe**2/tau**2)*campe
            SHy00 = math.exp(-temph**2/tau**2)*camph \
                / (self.zz0/math.sqrt(self.epsr[self.bgmater]))
        else:
            if tempe < 0.0:
                SEx00 = 0.0
            elif tempe < math.pi/omega_env:
                SEx00 = 0.5*(1.0 - math.cos(omega_env*tempe)) * campe
            else:
                SEx00 = campe

            if temph < 0.0:
                SHy00 = 0.0
            elif temph < math.pi/omega_env:
                SHy00 = 0.5*(1.0 - math.cos(omega_env*temph)) * camph \
                    / (self.zz0/math.sqrt(self.epsr[self.bgmater]))
            else:
                SHy00 = camph / (self.zz0/math.sqrt(self.epsr[self.bgmater]))
        
        # store source E field
        self.esource[jt] = SEx00

        # store source E and H fields for FFT
        if jt % self.sampint == 0:
            jfft = jt // self.sampint

        # Ex development
        for iz in range(1, self.mzz):
            imater = self.isdx[iz]
            self.SEx2[iz] = self.SEx1[iz] * self.ce1[imater] \
                - self.spx2[iz] * self.ce3[imater] \
                - (self.SHy1[iz] - self.SHy1[iz - 1]) * self.ckez[iz] * self.ce2[imater]
        
        # -z pml
        for iz in range(1, self.mz1):
            self.SpsiExz2m[iz] = self.SpsiExz1m[iz] * self.cbze[iz] \
                + (self.SHy1[iz] - self.SHy1[iz - 1]) * self.ccez[iz]
            self.SEx2[iz] = self.SEx2[iz] - self.SpsiExz2m[iz] * self.ce2[self.isdx[iz]]

        # +z pml
        for iz in range(self.mz2 + 1, self.mzz):
            izz = iz - self.mz2
            izzr = self.mzz - iz
            self.SpsiExz2p[izz] = self.SpsiExz1p[izz] * self.cbze[izzr] \
                + (self.SHy1[iz] - self.SHy1[iz - 1]) * self.ccze[izzr]
            self.SEx2[iz] = self.SEx2[iz] - self.SpsiExz2p[izz] * self.ce2[self.isdx[iz]]

        # source compensation for E
        self.SEx2[self.izst] = self.SEx2[self.izst] + self.cez2[self.isdx[self.izst]] * SHy00

        self.Sex2[0] = 0.0
        self.SEx2[self.mzz] = 0.0

        # Hy development
        for iz in range(self.mzz):
            self.SHy2[iz] = self.SHy1[iz] - (self.SEx2[iz + 1] - self.SEx2[iz]) * self.ckhz1[iz]

        # -z pml
        for iz in range(self.mz1):
            self.SpsiHyz2m[iz] = self.SpsiHyz1m[iz] * self.cbzh[iz] \
                + (self.SEx2[iz + 1] - self.SEx2[iz]) * self.cczh[iz]
            self.SHy2[iz] = self.SHy2[iz] - self.SpsiHyz2m[iz]*self.coefh

        # +z pml
        for iz in range(self.mz2 + 1, self.mzz):
            izz = iz - self.mz2
            izzr = self.mzz - iz - 1
            self.SpsiHyz2p[izz] = self.SpsiHyz1p[izz] * self.cbzh[izzr] \
                + (self.SEx2[iz + 1] - self.SEx2[iz]) * self.cczh[izzr]
            self.SHy2[iz] = self.SHy2[iz] - self.SpsiHyz2p[izz]*self.coefh

        # source compensation for H
        self.SHy2[self.izst-1] = self.SHy2[self.izst-1] + self.ckhz1[self.izst]*SEx00

        iy1 = self.mox1
        iy2 = self.moy2
        iz1 = self.moz1
        iz2 = self.moz2 - 1

        # -x boundary
        ix = self. mxx1
        for iz in range(iz1, iz2):
            self.Ez2[iz,iy1:iy2,ix] = self.Ez2[iz,iy1:iy2,ix] - self.cex2[self.isdz[iz]] * self.SHy2[iz]

        # +x boundary
        ix = self.mox2
        for iz in range(iz1, iz2):
            self.Ez2[iz,iy1:iy2,ix] = self.Ez2[iz,iy1:iy2,ix] + self.cex2[self.isdz[iz]] * self.SHy2[iz]

        ix1 = self.mox1
        ix2 = self.mox2 - 1

        # -z boundary
        iz = self.moz1
        self.Ex2[iz,iy1:iy2,ix1:ix2] = self.Ex2[iz,iy1:iy2,ix1:ix2] + self.cez2[self.isdx[iz]] * self.SHy2[iz-1]

        # +z boundary
        iz = self.moz2 - 1
        self.Ex2[iz,iy1:iy2,ix1:ix2] = self.Ex2[iz,iy1:iy2,ix1:ix2] - self.cez2[self.isdx[iz]] * self.SHy2[iz]

        # develop spx2 for ADE
        iz1 = 1
        iz2 = self.mzz
        self.spx2[iz1:iz2] = self.cj1[self.isdx[iz1:iz2]]*self.spx2[iz1:iz2] \
            + self.cj3[self.isdx[iz1:iz2]] * (self.SEx2[iz1:iz2] + self.SEx1[iz1:iz2])
        
        # update
        self.SEx1[:] = self.SEx2[:]
        self.SHy1[:] = self.SHy2[:]
        self.SpsiExz1m[:] = self.SpsiExz2m[:]
        self.SpsiExz1p[:] = self.SpsiExz2p[:]
        self.SpsiHyz1m[:] = self.SpsiHyz2m[:]
        self.SpsiHyz1p[:] = self.SpsiHyz2p[:]

    def normalinc_p_h(self):
        """ Source: x-polarized and z-propagating plane wave
            TF/SF compensation for H 
        """
        
        ix1 = self.mox1
        ix2 = self.mox2 - 1
        iz1 = self.moz1
        iz2 = self.moz2
        
        # -y boundary
        iy = self.my1 + self.msf
        for iz in range(iz1, iz2):
            for ix in range(ix1, ix2):
                self.Hz2[iz,iy-1,ix] = self.Hx2[iz,iy-1,ix] - self.ckhy1[iy-1] * self.SEx1[iz]

        # +y boundary
        iy = self.my2 - self.msf
        for iz in range(iz1, iz2):
            for ix in range(ix1, ix2):
                self.Hz2[iz,iy,ix] = self.Hz2[iz,iy,ix] + self.ckhy1[iy] * self.SEx1[iz]

        iy1 = self.moy1
        iy2 = self.moy2

        # -z boundary
        iz = self.moz1
        for iy in range(iy1, iy2):
            for ix in range(ix1, ix2):
                self.Hy2[iz-1,iy,ix] = self.Hy2[iz-1,iy,ix] + self.ckhz1[iz-1] * self.SEx1[iz]

        # +z boundary
        iz = self.moz2 - 1
        for iy in range(iy1, iy2):
            for ix in range(ix1, ix2):
                self.Hy2[iz,iy,ix] = self.Hy2[iz,iy,ix] - self.ckhz1[iz] * self.SEx1[iz]

    def sweep_isolate_h(self):
        ix1 = 0
        iy1 = 0
        iz1 = 0

        # Hx development
        ix2 = self.mxx + 1
        iy2 = self.myy
        iz2 = self.mzz

        for iy in range(iy1, iy2):
            self.Hx2[iz1:iz2,iy,ix1:ix2] = self.Hx1[iz1:iz2,iy,ix1:ix2] \
                - (self.Ez1[iz1:iz2,iy+1,ix1:ix2] - self.Ez2[iz1:iz2,iy,ix1:ix2]) * self.ckhy1[iy]
        for iz in range(iz1, iz2):
            self.Hx2[iz,iy1:iy2,ix1:ix2] = self.Hx2[iz,iy1:iy2,ix1:ix2] \
                + (self.Ey1[iz+1,iy1:iy2,ix1:ix2] - self.Ey2[iz,iy1:iy2,ix1:ix2]) * self.ckhz1[iz]
            
        # Hy development
        ix2 = self.mxx
        iy2 = self.myy + 1
        iz2 = self.mzz

        for iz in range(iz1, iz2):
            self.Hy2[iz,iy1:iy2,ix1:ix2] = self.Hy1[iz,iy1:iy2,ix1:ix2] \
                - (self.Ex2[iz+1,iy1:iy2,ix1:ix2] - self.Ex2[iz,iy1:iy2,ix1:ix2]) * self.ckhz1[iz]
        for ix in range(ix1, ix2):
            self.Hy2[iz1:iz2,iy1:iy2,ix] = self.Hy2[iz1:iz2,iy1:iy2,ix] \
                + (self.Ez2[iz1:iz2,iy1:iy2,ix+1] - self.Ez2[iz1:iz2,iy1:iy2,ix]) * self.ckhx1[ix]
            
        # Hz development
        ix2 = self.mxx
        iy2 = self.myy
        iz2 = self.mzz + 1

        for ix in range(ix1, ix2):
            self.Hz2[iz1:iz2,iy1:iy2,ix] = self.Hz1[iz1:iz2,iy1:iy2,ix] \
                - (self.Ey2[iz1:iz2,iy1:iy2,ix+1] - self.Ey2[iz1:iz2,iy1:iy2,ix]) * self.ckhx1[ix]
        for iy in range(iy1, iy2):
            self.Hz2[iz1:iz2,iy,ix1:ix2] = self.Hz2[iz1:iz2,iy,ix1:ix2] \
                + (self.Ex2[iz1:iz2,iy+1,ix1:ix2] - self.Ex2[iz1:iz2,iy,ix1:ix2]) * self.ckhy1[iy]
            
    def sweep_boundary_h(self):
        # -x boundary

        # Hy PML
        ix1 = 0
        iy1 = 0
        iz1 = 0
        ix2 = self.mx1
        iy2 = self.myy + 1
        iz2 = self.mzz

        for ix in range(ix1,ix2):
            self.psiHyx2m[iz1:iz2,iy1:iy2,ix] \
                = self.psiHyx1m[iz1:iz2,iy1:iy2,ix] * self.cbxh[ix] \
                + (self.Ez2[iz1:iz2,iy1:iy2,ix+1] - self.Ez2[iz1:iz2,iy1:iy2,ix]) * self.ccxh[ix]
            
            self.Hy2[iz1:iz2,iy1:iy2,ix] = self.Hy2[iz1:iz2,iy1:iy2,ix] \
                + self.psiHyx1m[iz1:iz2,iy1:iy2,ix] * self.coefh

        # Hz PML
        ix2 = self.mx1
        iy2 = self.myy
        iz2 = self.mzz + 1

        for ix in range(ix1,ix2):
            self.psiHzx2m[iz1:iz2,iy1:iy2,ix] \
                = self.psiHzx1m[iz1:iz2,iy1:iy2,ix] * self.cbxh[ix] \
                + (self.Ey2[iz1:iz2,iy1:iy2,ix+1] - self.Ey2[iz1:iz2,iy1:iy2,ix]) * self.ccxh[ix]
            
            self.Hz2[iz1:iz2,iy1:iy2,ix] = self.Hz2[iz1:iz2,iy1:iy2,ix] \
                - self.psiHzx1m[iz1:iz2,iy1:iy2,ix] * self.coefh
            
        # +x boundary

        # Hy PML
        ix1 = self.mx2
        iy1 = 0
        iz1 = 0
        ix2 = self.mxx
        iy2 = self.myy + 1
        iz2 = self.mzz

        for ix in range(ix1,ix2):
            ixx = ix - self.mx2
            ixxr = self.mxx - ix - 1
            self.psiHyx2p[iz1:iz2,iy1:iy2,ixx] \
                = self.psiHyx1p[iz1:iz2,iy1:iy2,ixx] * self.cbxh[ixxr] \
                + (self.Ez2[iz1:iz2,iy1:iy2,ix+1] - self.Ez2[iz1:iz2,iy1:iy2,ix]) * self.ccxh[ixxr]
            
            self.Hy2[iz1:iz2,iy1:iy2,ix] = self.Hy2[iz1:iz2,iy1:iy2,ix] \
                - self.psiHyx1p[iz1:iz2,iy1:iy2,ixx] * self.coefh
            
        # Hz PML
        ix2 = self.mxx
        iy2 = self.myy
        iz2 = self.mzz + 1

        for ix in range(ix1,ix2):
            ixx = ix - self.mx2
            ixxr = self.mxx - ix - 1
            self.psiHzx2p[iz1:iz2,iy1:iy2,ixx] \
                = self.psiHzx1p[iz1:iz2,iy1:iy2,ixx] * self.cbxh[ixxr] \
                + (self.Ey2[iz1:iz2,iy1:iy2,ix+1] - self.Ey2[iz1:iz2,iy1:iy2,ix]) * self.ccxh[ixxr]
            
            self.Hz2[iz1:iz2,iy1:iy2,ix] = self.Hz2[iz1:iz2,iy1:iy2,ix] \
                - self.psiHzx1p[iz1:iz2,iy1:iy2,ixx] * self.coefh
            
        # -y boundary

        # Hx PML
        ix1 = 0
        iy1 = 0
        iz1 = 0
        ix2 = self.mxx + 1
        iy2 = self.my1
        iz2 = self.mzz

        for iy in range(iy1,iy2):
            self.psiHxy2m[iz1:iz2,iy,ix1:ix2] \
                = self.psiHxy1m[iz1:iz2,iy,ix1:ix2] * self.cbyh[iy] \
                + (self.Ez2[iz1:iz2,iy+1,ix1:ix2] - self.Ez2[iz1:iz2,iy,ix1:ix2]) * self.ccyh[iy]
            
            self.Hx2[iz1:iz2,iy,ix1:ix2] = self.Hx2[iz1:iz2,iy,ix1:ix2] \
                - self.psiHxy1m[iz1:iz2,iy,ix1:ix2] * self.coefh
            
        # Hz PML
        ix2 = self.mxx
        iy2 = self.my1
        iz2 = self.mzz + 1

        for iy in range(iy1,iy2):
            self.psiHzy2m[iz1:iz2,iy,ix1:ix2] \
                = self.psiHzy1m[iz1:iz2,iy,ix1:ix2] * self.cbyh[iy] \
                + (self.Ex2[iz1:iz2,iy+1,ix1:ix2] - self.Ex2[iz1:iz2,iy,ix1:ix2]) * self.ccyh[iy]
            
            self.Hz2[iz1:iz2,iy,ix1:ix2] = self.Hz2[iz1:iz2,iy,ix1:ix2] \
                + self.psiHzy1m[iz1:iz2,iy,ix1:ix2] * self.coefh
            
        # +y boundary

        # Hx PML
        ix1 = 0
        iy1 = self.my2
        iz1 = 0
        ix2 = self.mxx + 1
        iy2 = self.myy
        iz2 = self.mzz

        for iy in range(iy1,iy2):
            iyy = iy - self.my2
            iyyr = self.myy - iy - 1
            self.psiHxy2p[iz1:iz2,iyy,ix1:ix2] \
                = self.psiHxy1p[iz1:iz2,iyy,ix1:ix2] * self.cbyh[iyyr] \
                + (self.Ez2[iz1:iz2,iy+1,ix1:ix2] - self.Ez2[iz1:iz2,iy,ix1:ix2]) * self.ccyh[iyyr]
            
            self.Hx2[iz1:iz2,iy,ix1:ix2] = self.Hx2[iz1:iz2,iy,ix1:ix2] \
                + self.psiHxy1p[iz1:iz2,iyy,ix1:ix2] * self.coefh
            
        # Hz PML
        ix2 = self.mxx
        iy2 = self.myy
        iz2 = self.mzz + 1

        for iy in range(iy1,iy2):
            iyy = iy - self.my2
            iyyr = self.myy - iy - 1
            self.psiHzy2p[iz1:iz2,iyy,ix1:ix2] \
                = self.psiHzy1p[iz1:iz2,iyy,ix1:ix2] * self.cbyh[iyyr] \
                + (self.Ex2[iz1:iz2,iy+1,ix1:ix2] - self.Ex2[iz1:iz2,iy,ix1:ix2]) * self.ccyh[iyyr]
            
            self.Hz2[iz1:iz2,iy,ix1:ix2] = self.Hz2[iz1:iz2,iy,ix1:ix2] \
                + self.psiHzy1p[iz1:iz2,iyy,ix1:ix2] * self.coefh
            
        # -z boundary

        # Hx PML
        ix1 = 0
        iy1 = 0
        iz1 = 0
        ix2 = self.mxx + 1
        iy2 = self.myy
        iz2 = self.mz1

        for iz in range(iz1,iz2):
            self.psiHxz2m[iz,iy1:iy2,ix1:ix2] \
                = self.psiHxz1m[iz,iy1:iy2,ix1:ix2] * self.cbzh[iz] \
                + (self.Ey2[iz+1,iy1:iy2,ix1:ix2] - self.Ey2[iz,iy1:iy2,ix1:ix2]) * self.cczh[iz]
            
            self.Hx2[iz,iy1:iy2,ix1:ix2] = self.Hx2[iz,iy1:iy2,ix1:ix2] \
                + self.psiHxz1m[iz,iy1:iy2,ix1:ix2] * self.coefh
        
        # Hy PML
        ix2 = self.mxx
        iy2 = self.myy + 1
        iz2 = self.mz1

        for iz in range(iz1,iz2):
            self.psiHyz2m[iz,iy1:iy2,ix1:ix2] \
                = self.psiHyz1m[iz,iy1:iy2,ix1:ix2] * self.cbzh[iz] \
                + (self.Ex2[iz+1,iy1:iy2,ix1:ix2] - self.Ex2[iz,iy1:iy2,ix1:ix2]) * self.cczh[iz]
            
            self.Hy2[iz,iy1:iy2,ix1:ix2] = self.Hy2[iz,iy1:iy2,ix1:ix2] \
                - self.psiHyz1m[iz,iy1:iy2,ix1:ix2] * self.coefh
            
        # +z boundary

        # Hx PML
        ix1 = 0
        iy1 = 0
        iz1 = self.mz2
        ix2 = self.mxx + 1
        iy2 = self.myy
        iz2 = self.mzz

        for iz in range(iz1,iz2):
            izz = iz - self.mz2
            izzr = self.mzz - iz - 1
            self.psiHxz2p[izz,iy1:iy2,ix1:ix2] \
                = self.psiHxz1p[izz,iy1:iy2,ix1:ix2] * self.cbzh[izzr] \
                + (self.Ey2[iz+1,iy1:iy2,ix1:ix2] - self.Ey2[iz,iy1:iy2,ix1:ix2]) * self.cczh[izzr]
            
            self.Hx2[iz,iy1:iy2,ix1:ix2] = self.Hx2[iz,iy1:iy2,ix1:ix2] \
                - self.psiHxz1p[izz,iy1:iy2,ix1:ix2] * self.coefh
            
        # Hy PML
        ix2 = self.mxx
        iy2 = self.myy + 1
        iz2 = self.mzz

        for iz in range(iz1,iz2):
            izz = iz - self.mz2
            izzr = self.mzz - iz - 1
            self.psiHyz2p[izz,iy1:iy2,ix1:ix2] \
                = self.psiHyz1p[izz,iy1:iy2,ix1:ix2] * self.cbzh[izzr] \
                + (self.Ex2[iz+1,iy1:iy2,ix1:ix2] - self.Ex2[iz,iy1:iy2,ix1:ix2]) * self.cczh[izzr]
            
            self.Hy2[iz,iy1:iy2,ix1:ix2] = self.Hy2[iz,iy1:iy2,ix1:ix2] \
                + self.psiHyz1p[izz,iy1:iy2,ix1:ix2] * self.coefh

    def sweep_isolate_e(self):

        ix2 = self.mxx
        iy2 = self.myy
        iz2 = self.mzz

        """----------------------
           Ex development
        ----------------------"""    

        ix1 = 0
        iy1 = 1
        iz1 = 1

        for iy in range(iy1, iy2):
            self.Ex2[iz1:iz2,iy,ix1:ix2] \
                = self.Ex1[iz1:iz2,iy,ix1:ix2] * self.ce1[self.idx[iz1:iz2,iy,ix1:ix2]] \
                - self.px2[iz1:iz2,iy,ix1:ix2] * self.ce3[self.idx[iz1:iz2,iy,ix1:ix2]] \
                + (self.Hz1[iz1:iz2,iy,ix1:ix2] - self.Hz1[iz1:iz2,iy-1,ix1:ix2]) \
                * self.ckey[iy] * self.ce2[self.idx[iz1:iz2,iy,ix1:ix2]]
            
        for iz in range(iz1, iz2):
            self.Ex2[iz,iy1:iy2,ix1:ix2] = self.Ex2[iz,iy1:iy2,ix1:ix2] \
                - (self.Hy1[iz,iy1:iy2,ix1:ix2] - self.Hy1[iz-1,iy1:iy2,ix1:ix2]) \
                * self.ckez[iz] * self.ce2[self.idx[iz,iy1:iy2,ix1:ix2]]
            
        """----------------------
           Ey development
        ----------------------"""

        ix1 = 1
        iy1 = 0
        iz1 = 1

        for iz in range(iz1, iz2):
            self.Ey2[iz,iy1:iy2,ix1:ix2] = self.Ey1[iz,iy1:iy2,ix1:ix2] * self.ce1[self.idy[iz,iy1:iy2,ix1:ix2]] \
                - self.py2[iz,iy1:iy2,ix1:ix2] * self.ce3[self.idy[iz,iy1:iy2,ix1:ix2]] \
                + (self.Hx1[iz,iy1:iy2,ix1:ix2] - self.Hx1[iz-1,iy1:iy2,ix1:ix2]) \
                * self.ckez[iz] * self.ce2[self.idy[iz,iy1:iy2,ix1:ix2]]
            
        for ix in range(ix1, ix2):
            self.Ey2[iz1:iz2,iy1:iy2,ix] = self.Ey2[iz1:iz2,iy1:iy2,ix] \
                - (self.Hz1[iz1:iz2,iy1:iy2,ix] - self.Hz1[iz1:iz2,iy1:iy2,ix-1]) \
                * self.ckex[ix] * self.ce2[self.idy[iz1:iz2,iy1:iy2,ix]]

        """----------------------
           Ez development
        ----------------------"""

        ix1 = 1
        iy1 = 1
        iz1 = 0

        for ix in range(ix1, ix2):
            self.Ez2[iz1:iz2,iy1:iy2,ix] = self.Ez1[iz1:iz2,iy1:iy2,ix] * self.ce1[self.idz[iz1:iz2,iy1:iy2,ix]] \
                - self.pz2[iz1:iz2,iy1:iy2,ix] * self.ce3[self.idz[iz1:iz2,iy1:iy2,ix]] \
                + (self.Hy1[iz1:iz2,iy1:iy2,ix] - self.Hy1[iz1:iz2,iy1:iy2,ix-1]) \
                * self.ckex[ix] * self.ce2[self.idz[iz1:iz2,iy1:iy2,ix]]
            
        for iy in range(iy1, iy2):
            self.Ez2[iz1:iz2,iy,ix1:ix2] = self.Ez2[iz1:iz2,iy,ix1:ix2] \
                - (self.Hx1[iz1:iz2,iy,ix1:ix2] - self.Hx1[iz1:iz2,iy-1,ix1:ix2]) \
                * self.ckey[iy] * self.ce2[self.idz[iz1:iz2,iy,ix1:ix2]]

    def sweep_boundary_e(self):

        ix2 = self.mxx
        iy2 = self.myy
        iz2 = self.mzz

        # -x-side boundary
        # Ey PML
        iy1 = 0
        iz1 = 1

        for ix in range(1, self.mx1):
            self.psiEyx2m[iz1:iz2,iy1:iy2,ix] = self.psiEyx1m[iz1:iz2,iy1:iy2,ix] * self.cbxe[ix] \
                + (self.Hz1[iz1:iz2,iy1:iy2,ix] - self.Hz1[iz1:iz2,iy1:iy2,ix-1]) * self.ccxe[ix]
            
            self.Ey2[iz1:iz2,iy1:iy2,ix] = self.Ey2[iz1:iz2,iy1:iy2,ix] \
                - self.psiEyx2m[iz1:iz2,iy1:iy2,ix] * self.ce2[self.idy[iz1:iz2,iy1:iy2,ix]]
            
        self.Ey2[:,:,0] = 0.0

        # Ez PML
        iy1 = 1
        iz1 = 0

        for ix in range(1, self.mx1):
            self.psiEzx2m[iz1:iz2,iy1:iy2,ix] = self.psiEzx1m[iz1:iz2,iy1:iy2,ix] * self.cbxe[ix] \
                + (self.Hy1[iz1:iz2,iy1:iy2,ix] - self.Hy1[iz1:iz2,iy1:iy2,ix-1]) * self.ccxe[ix]
            
            self.Ez2[iz1:iz2,iy1:iy2,ix] = self.Ez2[iz1:iz2,iy1:iy2,ix] \
                + self.psiEzx2m[iz1:iz2,iy1:iy2,ix] * self.ce2[self.idz[iz1:iz2,iy1:iy2,ix]]
            
        self.Ez2[:,:,0] = 0.0

        # +x-side boundary
        # Ey PML
        iy1 = 0
        iz1 = 1

        for ix in range(self.mx2 + 1, self.mxx):
            ixx = ix - self.mx2
            ixxr = self.mxx - ix
            self.psiEyx2p[iz1:iz2,iy1:iy2,ixx] = self.psiEyx1p[iz1:iz2,iy1:iy2,ixx] * self.cbxe[ixxr] \
                + (self.Hz1[iz1:iz2,iy1:iy2,ix] - self.Hz1[iz1:iz2,iy1:iy2,ix-1]) * self.ccxe[ixxr]
            
            self.Ey2[iz1:iz2,iy1:iy2,ix] = self.Ey2[iz1:iz2,iy1:iy2,ix] \
                - self.psiEyx2p[iz1:iz2,iy1:iy2,ixx] * self.ce2[self.idy[iz1:iz2,iy1:iy2,ix]]
            
        self.Ey2[:,:,self.mxx] = 0.0

        # Ez PML
        iy1 = 1
        iz1 = 0

        for ix in range(self.mx2 + 1, self.mxx):
            ixx = ix - self.mx2
            ixxr = self.mxx - ix
            self.psiEzx2p[iz1:iz2,iy1:iy2,ixx] = self.psiEzx1p[iz1:iz2,iy1:iy2,ixx] * self.cbxe[ixxr] \
                + (self.Hy1[iz1:iz2,iy1:iy2,ix] - self.Hy1[iz1:iz2,iy1:iy2,ix-1]) * self.ccxe[ixxr]
            
            self.Ez2[iz1:iz2,iy1:iy2,ix] = self.Ez2[iz1:iz2,iy1:iy2,ix] \
                + self.psiEzx2p[iz1:iz2,iy1:iy2,ixx] * self.ce2[self.idz[iz1:iz2,iy1:iy2,ix]]
            
        self.Ez2[:,:,self.mxx] = 0.0

        # -y-side boundary
        # Ex PML
        ix1 = 0
        iz1 = 1

        for iy in range(1, self.my1):
            self.psiExy2m[iz1:iz2,iy,ix1:ix2] = self.psiExy1m[iz1:iz2,iy,ix1:ix2] * self.cbye[iy] \
                + (self.Hz1[iz1:iz2,iy,ix1:ix2] - self.Hz1[iz1:iz2,iy-1,ix1:ix2]) * self.ccye[iy]
            
            self.Ex2[iz1:iz2,iy,ix1:ix2] = self.Ex2[iz1:iz2,iy,ix1:ix2] \
                + self.psiExy2m[iz1:iz2,iy,ix1:ix2] * self.ce2[self.idx[iz1:iz2,iy,ix1:ix2]]
            
        self.Ex2[:,0,:] = 0.0

        # Ez PML
        ix1 = 1
        iz1 = 0

        for iy in range(1, self.my1):
            self.psiEzy2m[iz1:iz2,iy,ix1:ix2] = self.psiEzy1m[iz1:iz2,iy,ix1:ix2] * self.cbye[iy] \
                + (self.Hx1[iz1:iz2,iy,ix1:ix2] - self.Hx1[iz1:iz2,iy-1,ix1:ix2]) * self.ccye[iy]
            
            self.Ez2[iz1:iz2,iy,ix1:ix2] = self.Ez2[iz1:iz2,iy,ix1:ix2] \
                - self.psiEzy2m[iz1:iz2,iy,ix1:ix2] * self.ce2[self.idz[iz1:iz2,iy,ix1:ix2]]
            
        self.Ez2[:,0,:] = 0.0

        # +y-side boundary
        # Ex PML
        ix1 = 0
        iz1 = 1

        for iy in range(self.my2 + 1, self.myy):
            iyy = iy - self.my2
            iyyr = self.myy - iy
            self.psiExy2p[iz1:iz2,iyy,ix1:ix2] = self.psiExy1p[iz1:iz2,iyy,ix1:ix2] * self.cbye[iyyr] \
                + (self.Hz1[iz1:iz2,iy,ix1:ix2] - self.Hz1[iz1:iz2,iy-1,ix1:ix2]) * self.ccye[iyyr]
            
            self.Ex2[iz1:iz2,iy,ix1:ix2] = self.Ex2[iz1:iz2,iy,ix1:ix2] \
                + self.psiExy2p[iz1:iz2,iyy,ix1:ix2] * self.ce2[self.idx[iz1:iz2,iy,ix1:ix2]]
            
        self.Ex2[:,self.myy,:] = 0.0

        # Ez PML
        ix1 = 1
        iz1 = 0

        for iy in range(self.my2 + 1, self.myy):
            iyy = iy - self.my2
            iyyr = self.myy - iy
            self.psiEzy2p[iz1:iz2,iyy,ix1:ix2] = self.psiEzy1p[iz1:iz2,iyy,ix1:ix2] * self.cbye[iyyr] \
                + (self.Hx1[iz1:iz2,iy,ix1:ix2] - self.Hx1[iz1:iz2,iy-1,ix1:ix2]) * self.ccye[iyyr]
            
            self.Ez2[iz1:iz2,iy,ix1:ix2] = self.Ez2[iz1:iz2,iy,ix1:ix2] \
                - self.psiEzy2p[iz1:iz2,iyy,ix1:ix2] * self.ce2[self.idz[iz1:iz2,iy,ix1:ix2]]

        self.Ez2[:,self.myy,:] = 0.0

        # -z-side boundary
        # Ex PML
        ix1 = 0
        iy1 = 1

        for iz in range(1, self.mz1):
            self.psiExz2m[iz,iy1:iy2,ix1:ix2] = self.psiExz1m[iz,iy1:iy2,ix1:ix2] * self.cbze[iz] \
                + (self.Hy1[iz,iy1:iy2,ix1:ix2] - self.Hy1[iz-1,iy1:iy2,ix1:ix2]) * self.ccze[iz]
            
            self.Ex2[iz,iy1:iy2,ix1:ix2] = self.Ex2[iz,iy1:iy2,ix1:ix2] \
                - self.psiExz2m[iz,iy1:iy2,ix1:ix2] * self.ce2[self.idx[iz,iy1:iy2,ix1:ix2]]
            
        self.Ex2[0,:,:] = 0.0

        # Ey PML
        ix1 = 1
        iy1 = 0

        for iz in range(1, self.mz1):
            self.psiEyz2m[iz,iy1:iy2,ix1:ix2] = self.psiEyz1m[iz,iy1:iy2,ix1:ix2] * self.cbze[iz] \
                + (self.Hx1[iz,iy1:iy2,ix1:ix2] - self.Hx1[iz-1,iy1:iy2,ix1:ix2]) * self.ccze[iz]
            
            self.Ey2[iz,iy1:iy2,ix1:ix2] = self.Ey2[iz,iy1:iy2,ix1:ix2] \
                + self.psiEyz2m[iz,iy1:iy2,ix1:ix2] * self.ce2[self.idy[iz,iy1:iy2,ix1:ix2]]
            
        self.Ey2[0,:,:] = 0.0

        # +z-side boundary
        # Ex PML
        ix1 = 0
        iy1 = 1

        for iz in range(self.mz2 + 1, self.mzz):
            izz = iz - self.mz2
            izzr = self.mzz - iz
            self.psiExz2p[izz,iy1:iy2,ix1:ix2] = self.psiExz1p[izz,iy1:iy2,ix1:ix2] * self.cbze[izzr] \
                + (self.Hy1[iz,iy1:iy2,ix1:ix2] - self.Hy1[iz-1,iy1:iy2,ix1:ix2]) * self.ccze[izzr]
            
            self.Ex2[iz,iy1:iy2,ix1:ix2] = self.Ex2[iz,iy1:iy2,ix1:ix2] \
                - self.psiExz2p[izz,iy1:iy2,ix1:ix2] * self.ce2[self.idx[iz,iy1:iy2,ix1:ix2]]
            
        self.Ex2[self.mzz,:,:] = 0.0
            
        # Ey PML
        ix1 = 1
        iy1 = 0

        for iz in range(self.mz2 + 1, self.mzz):
            izz = iz - self.mz2
            izzr = self.mzz - iz
            self.psiEyz2p[izz,iy1:iy2,ix1:ix2] = self.psiEyz1p[izz,iy1:iy2,ix1:ix2] * self.cbze[izzr] \
                + (self.Hx1[iz,iy1:iy2,ix1:ix2] - self.Hx1[iz-1,iy1:iy2,ix1:ix2]) * self.ccze[izzr]
            
            self.Ey2[iz,iy1:iy2,ix1:ix2] = self.Ey2[iz,iy1:iy2,ix1:ix2] \
                + self.psiEyz2p[izz,iy1:iy2,ix1:ix2] * self.ce2[self.idy[iz,iy1:iy2,ix1:ix2]]
            
        self.Ey2[self.mzz,:,:] = 0.0

    def develop_pcurrent(self):
        ix2 = self.mxx
        iy2 = self.myy
        iz2 = self.mzz

        # px2 development
        ix1 = 0
        iy1 = 1
        iz1 = 1

        self.px2[iz1:iz2,iy1:iy2,ix1:ix2] \
            = self.cj1[self.idx[iz1:iz2,iy1:iy2,ix1:ix2]] * self.px2[iz1:iz2,iy1:iy2,ix1:ix2] \
            + self.cj3[self.idx[iz1:iz2,iy1:iy2,ix1:ix2]] * (self.Ex2[iz1:iz2,iy1:iy2,ix1:ix2] + self.Ex1[iz1:iz2,iy1:iy2,ix1:ix2])
        
        # py2 development
        ix1 = 1
        iy1 = 0
        iz1 = 1

        self.py2[iz1:iz2,iy1:iy2,ix1:ix2] \
            = self.cj1[self.idy[iz1:iz2,iy1:iy2,ix1:ix2]] * self.py2[iz1:iz2,iy1:iy2,ix1:ix2] \
            + self.cj3[self.idy[iz1:iz2,iy1:iy2,ix1:ix2]] * (self.Ey2[iz1:iz2,iy1:iy2,ix1:ix2] + self.Ey1[iz1:iz2,iy1:iy2,ix1:ix2])

        # pz2 development
        ix1 = 1
        iy1 = 1
        iz1 = 0

        self.pz2[iz1:iz2,iy1:iy2,ix1:ix2] \
            = self.cj1[self.idz[iz1:iz2,iy1:iy2,ix1:ix2]] * self.pz2[iz1:iz2,iy1:iy2,ix1:ix2] \
            + self.cj3[self.idz[iz1:iz2,iy1:iy2,ix1:ix2]] * (self.Ez2[iz1:iz2,iy1:iy2,ix1:ix2] + self.Ez1[iz1:iz2,iy1:iy2,ix1:ix2])
        
    def update_field(self):

        self.Ex1[:,:,:] = self.Ex2[:,:,:]
        self.Ey1[:,:,:] = self.Ey2[:,:,:]
        self.Ez1[:,:,:] = self.Ez2[:,:,:]
        self.Hx1[:,:,:] = self.Hx2[:,:,:]
        self.Hy1[:,:,:] = self.Hy2[:,:,:]
        self.Hz1[:,:,:] = self.Hz2[:,:,:]

        self.psiEzx1m[:,:,:] = self.psiEzx2m[:,:,:]
        self.psiEyx1m[:,:,:] = self.psiEyx2m[:,:,:]
        self.psiHzx1m[:,:,:] = self.psiHzx2m[:,:,:]
        self.psiHyx1m[:,:,:] = self.psiHyx2m[:,:,:]
        self.psiHzx1p[:,:,:] = self.psiHzx2p[:,:,:]
        self.psiHyx1p[:,:,:] = self.psiHyx2p[:,:,:]
        self.psiEzx1p[:,:,:] = self.psiEzx2p[:,:,:]
        self.psiEyx1p[:,:,:] = self.psiEyx2p[:,:,:]

        self.psiEzy1m[:,:,:] = self.psiEzy2m[:,:,:]
        self.psiExy1m[:,:,:] = self.psiExy2m[:,:,:]
        self.psiHzy1m[:,:,:] = self.psiHzy2m[:,:,:]
        self.psiHxy1m[:,:,:] = self.psiHxy2m[:,:,:]
        self.psiEzy1p[:,:,:] = self.psiEzy2p[:,:,:]
        self.psiExy1p[:,:,:] = self.psiExy2p[:,:,:]
        self.psiHzy1p[:,:,:] = self.psiHzy2p[:,:,:]
        self.psiHxy1p[:,:,:] = self.psiHxy2p[:,:,:]

        self.psiEyz1m[:,:,:] = self.psiEyz2m[:,:,:]
        self.psiExz1m[:,:,:] = self.psiExz2m[:,:,:]
        self.psiHyz1m[:,:,:] = self.psiHyz2m[:,:,:]
        self.psiHxz1m[:,:,:] = self.psiHxz2m[:,:,:]
        self.psiEyz1p[:,:,:] = self.psiEyz2p[:,:,:]
        self.psiExz1p[:,:,:] = self.psiExz2p[:,:,:]
        self.psiHyz1p[:,:,:] = self.psiHyz2p[:,:,:]
        self.psiHxz1p[:,:,:] = self.psiHxz2p[:,:,:]

    def save_idv(self):
        """ save material index distribution """

        for epsmon in self.iepsmons:
            if epsmon.pol == 'x':
                if epsmon.axis == 'x': # normal to x-axis
                    ieps2d = self.idx[:self.mzz+1, :self.myy+1, epsmon.position]
                elif epsmon.axis == 'y': # normal to y-axis
                    ieps2d = self.idx[:self.mzz+1, epsmon.position, :self.mxx]
                else:
                    ieps2d = self.idx[epsmon.position, :self.myy+1, :self.mxx]
            elif epsmon.pol == 'y':
                if epsmon.axis == 'x':
                    ieps2d = self.idy[:self.mzz+1, :self.myy, epsmon.position]
                elif epsmon.axis == 'y':
                    ieps2d = self.idy[:self.mzz+1, epsmon.position, :self.mxx+1]
                else:
                    ieps2d = self.idy[epsmon.position, :self.myy, :self.mxx+1]
            else:
                if epsmon.axis == 'x':
                    ieps2d = self.idz[:self.mzz, :self.myy+1, epsmon.position]
                elif epsmon.axis == 'y':
                    ieps2d = self.idz[:self.mzz, epsmon.position, :self.mxx+1]
                else:
                    ieps2d = self.idz[epsmon.position, :self.myy+1, :self.mxx+1]
            
            if not os.path.exists('./field'):
                os.makedirs('./field')
            np.savetxt(epsmon.fname, ieps2d, fmt = '%d', delimiter = ' ')

    def save_ehfield(self, numt):
        """ save electric field and magnetic field """

        for ifieldmon in self.ifieldmons:
            location = ifieldmon.position
            ehfield = ifieldmon.ehfield

            # normal to x-axis
            if ifieldmon.axis == 'x':
                if ehfield == 'Ex':
                    field2d = self.Ex2[0:self.mzz+1, 0:self.myy+1, location]
                elif ehfield == 'Ey':
                    field2d = self.Ey2[0:self.mzz+1, 0:self.myy, location]
                elif ehfield == 'Ez':
                    field2d = self.Ez2[0:self.mzz, 0:self.myy+1, location]
                elif ehfield == 'Hx':
                    field2d = self.Hx2[0:self.mzz, 0:self.myy, location]
                elif ehfield == 'Hy':
                    field2d = self.Hy2[0:self.mzz, 0:self.myy+1, location]
                elif ehfield == 'Hz':
                    field2d = self.Hz2[0:self.mzz+1, 0:self.myy, location]

            # normal to y-axis
            elif ifieldmon.axis == 'y':
                if ehfield == 'Ex':
                    field2d = self.Ex2[0:self.mzz+1, location, 0:self.mxx]
                elif ehfield == 'Ey':
                    field2d = self.Ey2[0:self.mzz+1, location, 0:self.mxx+1]
                elif ehfield == 'Ez':
                    field2d = self.Ez2[0:self.mzz, location, 0:self.mxx+1]
                elif ehfield == 'Hx':
                    field2d = self.Hx2[0:self.mzz, location, 0:self.mxx+1]
                elif ehfield == 'Hy':
                    field2d = self.Hy2[0:self.mzz, location, 0:self.mxx]
                elif ehfield == 'Hz':
                    field2d = self.Hz1[0:self.mzz+1, location, 0:self.mxx]  # strange but Hz1 is used

            # normal to z-axis
            elif ifieldmon.axis == 'z':
                if ehfield == 'Ex':
                    field2d = self.Ex2[location, 0:self.myy+1, 0:self.mxx]
                elif ehfield == 'Ey':
                    field2d = self.Ey2[location, 0:self.myy, 0:self.mxx+1]
                elif ehfield == 'Ez':
                    field2d = self.Ez2[location, 0:self.myy+1, 0:self.mxx+1]
                elif ehfield == 'Hx':
                    field2d = self.Hx2[location, 0:self.myy, 0:self.mxx+1]
                elif ehfield == 'Hy':
                    field2d = self.Hy2[location, 0:self.myy+1, 0:self.mxx]
                elif ehfield == 'Hz':
                    field2d = self.Hz2[location, 0:self.myy, 0:self.mxx]

            if not os.path.exists('./field'):
                os.makedirs('./field')
            fname = ifieldmon.prefix + '{0:0>3}'.format(numt) + '.txt'
            np.savetxt(fname, field2d, fmt = '%.6e', delimiter = ' ')

    def detect_efield(self, jt):
        """ detection of E field """

        for i, detector in enumerate(self.idetectors):
            ix = detector.x
            iy = detector.y
            iz = detector.z

            if detector.pol == 'x':
                self.edetect[i][jt] = self.Ex1[iz,iy,ix]
            elif detector.pol == 'y':
                self.edetect[i][jt] = self.Ey1[iz,iy,ix]
            else:
                self.edetect[i][jt] = self.Ez1[iz,iy,ix]

    def detect_spectra(self):
        """ Fourier Transformation to obtain E-field spectra """

        if not os.path.exists('./field'):
            os.makedirs('./field')
        fname = 'field/Response.txt'
        col = 'Time(ps) Source'
        for i in range(len(self.idetectors)):
            col += ' Detector[' + str(i) + ']'
        atime = np.arange(0, self.mt) * self.dt * 1.0e12
        atime = np.append([atime], [self.esource], axis=0)
        atime = np.append(atime, self.edetect, axis=0)
        np.savetxt(fname, atime.T, fmt = '%.6e', delimiter = ' ', header = col, comments='')

        esource2 = self.esource[::self.sampint]
        esourceft = np.absolute(np.fft.rfft(esource2, n=self.mfft2))**2
        edetect2 = self.edetect[:,::self.sampint]
        edetectft = np.absolute(np.fft.rfft(edetect2, n=self.mfft2, axis=1))**2
        col = 'Frequency(THz) Wavelength(um) Source'
        for i in range(len(self.idetectors)):
            col += ' Detector[' + str(i) + ']'
        thz = np.arange(self.mfft2//2+1, dtype=np.float64) * 1.0e-12 / (self.mfft2 * self.dt * self.sampint)
        wavelength = np.ones(self.mfft2//2+1) * self.cc * 1.0e-6
        wavelength[1:] = wavelength[1:] / thz[1:]
        wavelength[0] = wavelength[1]
        thz = np.append([thz], [wavelength], axis=0)
        thz = np.append(thz, [esourceft], axis=0)
        thz = np.append(thz, edetectft, axis=0)
        if not os.path.exists('./field'):
            os.makedirs('./field')
        fname = './field/Spectra.txt'
        np.savetxt(fname, thz.T, fmt = '%.6e', delimiter = ' ', header = col, comments='')





        









    



