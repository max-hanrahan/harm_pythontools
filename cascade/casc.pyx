#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#cython: cdivision_warnings=False
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
cimport openmp
from libc.math cimport log, exp, sqrt, pow, fabs
from libc.stdlib cimport malloc, free
from cpython.exc cimport PyErr_CheckSignals
from cython.parallel import parallel, prange

DTYPE = np.float64
ctypedef np.float_t DTYPE_t

cdef double tiny = 1e-300

def fg_p( Eg not None, Ee not None, SeedPhoton seed not None):
    return fgvec( Eg, Ee, seed )

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef np.ndarray[double, ndim=1] fgvec( np.ndarray[double, ndim=1] Eg, np.ndarray[double, ndim=1] Ee, SeedPhoton seed):
    cdef int i
    cdef int dim = Ee.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] Eg1 = np.zeros_like(Ee)
    for i from 0 <= i < dim:
        Eg1[i] = fg( Eg[i], Ee[i], seed )
    return( Eg1 )

cdef inline double fg( double Eg, double Ee, SeedPhoton seed) nogil:
    cdef double Ep = Ee-Eg
    cdef double fgval = ( (seed.f(Eg/(2*Ee*Ep))/(2*Ep*Ep)) if (Ep>0 and Ee>0 and Eg>0) else (0) )
    return( fgval )

cdef inline double K1( double Enew, double Eold, SeedPhoton seed ) nogil:
    cdef double K = (4*fg(2*Enew,Eold,seed)) #if (2*Enew>=seed.Egmin) else (0)
    return( K )

cdef inline double K2( double Enew, double Eold, SeedPhoton seed ) nogil:
    cdef double K = fg(Eold-Enew,Eold,seed) if Eold > Enew else 0
    return( K )

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef public double* get_data( np.ndarray[double, ndim=1] nparray ):
    return <double *>nparray.data

def flnew( flold not None, flold_rad not None, flnew not None, flnew_rad not None, seed not None, grid not None, altgrid not None,  do_enforce_energy_conservation not None ):
    return flnew_c( flold, flold_rad, flnew, flnew_rad, seed, grid, altgrid, do_enforce_energy_conservation )

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef void compute_inner_convolution_c( int i,
                                     SeedPhoton seed, 
                                     Func flold_func, 
                                     Grid grid,
                                     Grid altgrid,
                                     double* ptemp0,
                                     double* ptemp0b,
                                     double* ptemp1,
                                     double* ptemp1b,
                                     double* ptemp2,
                                     double* ptemp2b) nogil:
    cdef double temp1, temp1b, temp2, temp2b
    cdef double *Evec_data = grid.Egrid_data
    cdef double *Evecb_data = altgrid.Egrid_data
    cdef double *flold_data = flold_func.func_vec_data
    cdef int dim1 = flold_func.Ngrid
    cdef double Eenew = Evec_data[i]
    #cdef int dim2b = altgrid.Ngrid
    temp1 = 0
    temp1b = 0
    temp2 = 0
    temp2b = 0
    for j from 0 <= j < dim1: #in xrange(dim1):
        temp1 += K1(Eenew,Evec_data[j],seed)*flold_data[j]*grid.dEdxgrid_data[j]*grid.dx
        temp1b += K1(Eenew,Evecb_data[j],seed)*flold_func.fofE(Evecb_data[j])*altgrid.dEdxgrid_data[j]*altgrid.dx
        temp2 += K2(Eenew,Eenew+Evec_data[j],seed)*flold_func.fofE(Eenew+Evec_data[j])*grid.dEdxgrid_data[j]*grid.dx
        temp2b += K2(Eenew,Eenew+Evecb_data[j],seed)*flold_func.fofE(Eenew+Evecb_data[j])*altgrid.dEdxgrid_data[j]*altgrid.dx
    # temp1sum += temp1*grid.dEdxgrid_data[i]*grid.dx
    #combine the two integrals into one more accurate integral (with twice res)
    #lepton energy lost to non-pair-producing gamma-rays
    ptemp0[0] = temp1 if (2*Eenew<seed.Egmin) else (0)
    ptemp0b[0] = temp1b if (2*Eenew<seed.Egmin) else (0)
    #lepton energy gained via pair-producing gamma-rays
    ptemp1[0] = temp1 if (2*Eenew>=seed.Egmin) else (0)
    ptemp1b[0] = temp1b if (2*Eenew>=seed.Egmin) else (0)
    #lepton energy gained/lost from IC cooling
    ptemp2[0] = temp2
    ptemp2b[0] = temp2b
    return

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef double flnew_c( Func flold_func, Func flold_rad_func, Func flnew_func, Func flnew_rad_func, SeedPhoton seed, Grid grid, Grid altgrid, int do_enforce_energy_conservation ) except *:
    """Expect E and flold defined on a regular log grid, Evec"""
    cdef int i
    cdef int j
    cdef double a, b, c, d, delta
    #cdef Grid grid = flold_func
    #cdef Grid altgrid = Grid(grid.Emin, grid.Emax, grid.E0, grid.Ngrid*2, di = grid.di)
    cdef double *Evec_data = grid.Egrid_data
    cdef double *Evecb_data = altgrid.Egrid_data
    # cdef double *flnew_data = flnew_func.func_vec_data
    # cdef double *lflnew_data = flnew_func.lfunc_vec_data
    cdef double *flold_data = flold_func.func_vec_data
    cdef double *flold_rad_data = flold_rad_func.func_vec_data
    cdef int dim1 = flold_func.Ngrid
    cdef int dim2b = altgrid.Ngrid
    cdef double E1, E2, N1, N2, Nold, dE1, dE2, dE, dN1, dN2, dN
    cdef double nw1, nw2, nwnorm, ew1, ew2, ewnorm
    cdef double *flnew_rad_data =  <double *>malloc(dim1 * sizeof(double))
    cdef double *flnew_rad_alt_data =  <double *>malloc(dim1 * sizeof(double))
    cdef double *flnew_gg_data =  <double *>malloc(dim1 * sizeof(double))
    cdef double *flnew_gg_alt_data =  <double *>malloc(dim1 * sizeof(double))
    cdef double *flnew_ic_data =  <double *>malloc(dim1 * sizeof(double))
    cdef double *flnew_ic_alt_data =  <double *>malloc(dim1 * sizeof(double))
    cdef double temp0, temp0b, temp1, temp1b, temp2, temp2b
    #
    
    #old number of electrons
    Nold = flold_func.norm()

    N1 = 0
    N2 = 0
    #for i from 0 <= i < dim1: 
    for i in prange(dim1, nogil=True):
        #reset them to zero so that they are local
        temp0 = 0
        temp0b = 0
        temp1 = 0
        temp1b = 0
        temp2 = 0
        temp2b = 0
        compute_inner_convolution_c(i, seed, flold_func, grid, altgrid, &temp0, &temp0b, &temp1, &temp1b, &temp2, &temp2b)
        #radiated away via gamma
        flnew_rad_data[i] = temp0
        flnew_rad_alt_data[i] = temp0b
        #put back into electrons via gamma pair production
        flnew_gg_data[i] = temp1
        flnew_gg_alt_data[i] = temp1b
        #redistribution of lepton energy via IC
        flnew_ic_data[i] = temp2
        flnew_ic_alt_data[i] = temp2b
        N1 += flnew_ic_data[i]    *grid.dEdxgrid_data[i]*grid.dx
        N2 += flnew_ic_alt_data[i]*grid.dEdxgrid_data[i]*grid.dx

    dN1 = N1 - Nold
    dN2 = N2 - Nold

    #print( "%e" % dE1 )

    ###########################################################################################
    #
    # Conserve the number of electrons in IC cooling: only affects temp2 and temp2b
    #
    if dN1 < 0 and dN2 > 0 or dN1 > 0 and dN2 < 0 or fabs(dN1) > 2*fabs(dN2) or fabs(dN2) > 2*fabs(dN1):
        nwnorm = dN2 - dN1
        nw1 =  dN2 / nwnorm
        nw2 = -dN1 / nwnorm
    elif fabs(dN1) < fabs(dN2):
        nw1 = 1
        nw2 = 0
    else:
        nw1 = 0
        nw2 = 1

    #ensure lepton number conservation under IC cooling
    for i from 0 <= i < dim1:
        flnew_ic_data[i] = nw1 * flnew_ic_data[i] + nw2 * flnew_ic_alt_data[i]
    #
    #
    ###########################################################################################

    if do_enforce_energy_conservation:
        #now that ic_data is finalized, vary the other two: rad_data and gg_data 
        E1 = 0
        E2 = 0
        for i from 0 <= i < dim1:
            E1 += (flnew_rad_data[i]    +flnew_gg_data[i]    +flnew_ic_data[i])*Evec_data[i]*grid.dEdxgrid_data[i]*grid.dx
            E2 += (flnew_rad_alt_data[i]+flnew_gg_alt_data[i]+flnew_ic_data[i])*Evec_data[i]*grid.dEdxgrid_data[i]*grid.dx

        #old energy (before scattering; do not count energy of escaped, non-pair-producing gamma-rays)
        Eold = flold_func.Etot() #+flold_rad_func.Etot()

        #changes in energy (should be zero so energy is conserved)
        dE1 = E1 - Eold
        dE2 = E2 - Eold


        ###########################################################################################
        #
        # Conserve total energy in gamma pair production: only affects temp0, temp0b, temp1 and temp1b
        #
        if dE1 < 0 and dE2 > 0 or dE1 > 0 and dE2 < 0 or fabs(dE1) > 1.5*fabs(dE2) or fabs(dE2) > 1.5*fabs(dE1):
            ewnorm = dE2 - dE1
            ew1 =  dE2 / ewnorm
            ew2 = -dE1 / ewnorm
        elif fabs(dE1) < fabs(dE2):
            ew1 = 1
            ew2 = 0
        else:
            ew1 = 0
            ew2 = 1

        #print( "dE1 = %e, dE2 = %e, dE = %e" % (dE1, dE2, ew1*dE1+ew2*dE2) )

        #ensure energy conservation under pair production
        for i from 0 <= i < dim1:
            flnew_rad_data[i] = ew1 * flnew_rad_data[i] + ew2 * flnew_rad_alt_data[i]
            flnew_gg_data[i]  = ew1 * flnew_gg_data[i]  + ew2 * flnew_gg_alt_data[i]
        #
        #
        ###########################################################################################

    for i from 0 <= i < dim1:
        #the result should conserve both number and energy of electrons
        flnew_func.set_funci_c(i,flnew_gg_data[i] + flnew_ic_data[i])
        #add non-pair-producing, radiated emission to the one from previous time step to keep track of total radiation
        flnew_rad_func.set_funci_c(i,flold_rad_data[i]+flnew_rad_data[i])

    # dN = w1*dN1+w2*dN2
    # if 1 or fabs(dN) > 1e-2:
    #     print( "Jump in the number of electrons: dN = %g, dN1 = %g, dN2 = %g, w1 = %g, w2 = %g" % (dN, dN1, dN2, w1, w2) )
    #     for i from 0 <= i < dim1:
    #         print( i, w1*deltaN1[i]+w2*deltaN2[i] )
    free(flnew_rad_alt_data)
    free(flnew_rad_data)
    free(flnew_gg_alt_data)
    free(flnew_gg_data)
    free(flnew_ic_alt_data)
    free(flnew_ic_data)

    #this is supposed to pass KeyboardInterrupt signal and other signals to python, but it does not do that
    PyErr_CheckSignals()

    return(nw1*N1+nw2*N2)

###############################
#
#  CLASSES
#
###############################        
    

###############################
#
#  SEED PHOTON
#
###############################        

cdef public class SeedPhoton [object CSeedPhoton, type TSeedPhoton ]:
    """our seed photon class"""
    cdef public double Emin
    cdef public double Emax
    cdef public double s
    cdef public double Egmin
    cdef public double Nprefactor

    def __init__(self, double Emin, double Emax, double s):
        self.Emin = Emin
        self.Emax = Emax
        self.s = s
        #minimum energy gamma-ray to be able to pair produce
        self.Egmin = 2./Emax
        self.Nprefactor = (1.-s)/(pow(Emax,1-s)-pow(Emin,1-s))

    cpdef int canPairProduce(self, double E):
        return( E > self.Egmin )

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef np.ndarray[double, ndim=1] f_vec(self, np.ndarray[double, ndim=1] E):
        cdef int i
        cdef np.ndarray[double, ndim=1] f_out = np.zeros_like(E)
        for i from 0 <= i < E.shape[0]:
            f_out[i] = self.f(E[i])
        return f_out

    cdef double f(self, double E) nogil:
        return( self.Nprefactor*E**(-self.s) if (E >= self.Emin and E <= self.Emax) else 0 )

    cpdef double minEg(self, double Eenew, double grid_Emin):
        """ Returns minimum gamma-ray energy """
        cdef double bottom = 1-2*self.Emax*Eenew
        cdef double minEg_val
        if bottom > 0:
            minEg_val = 2*self.Emin*Eenew**2/bottom
            return minEg_val if minEg_val > grid_Emin else grid_Emin
        else:
            return grid_Emin

    cpdef double maxEg(self, double Eenew, double grid_Emax):
        """ Returns minimum gamma-ray energy """
        cdef double bottom = 1-2*self.Emax*Eenew
        cdef double maxEg_val
        if bottom > 0:
            maxEg_val = 2*self.Emax*Eenew**2/bottom
            return maxEg_val if maxEg_val < grid_Emax else grid_Emax
        else:
            return grid_Emax


    cpdef double minEg1(self, double Eenew, double grid_Emin):
        """ Returns minimum energy electron contributing to Eenew"""
        cdef double minEg_val
        minEg_val = Eenew*(1+sqrt(1+1/(self.Emax*Eenew)))
        return minEg_val if minEg_val > grid_Emin else grid_Emin

    cpdef double maxEg1(self, double Eenew, double grid_Emax):
        """ Returns maximum energy electron contributing to Eenew"""
        cdef double maxEg_val
        maxEg_val = Eenew*(1+sqrt(1+1/(self.Emin*Eenew)))
        return maxEg_val if maxEg_val < grid_Emax else grid_Emax

    cpdef double minEg2(self, double Eenew, double grid_Emin):
        """ Returns minimum energy electron contributing to Eenew"""
        cdef double bottom = 1-2*self.Emin*Eenew
        cdef double minEg_val
        if bottom > 0:
            minEg_val = Eenew/bottom
            return minEg_val if minEg_val > grid_Emin else grid_Emin
        else:
            return grid_Emin

    cpdef double maxEg2(self, double Eenew, double grid_Emax):
        """ Returns maximum energy electron contributing to Eenew"""
        cdef double bottom = 1-2*self.Emax*Eenew
        cdef double maxEg_val
        if bottom > 0:
            maxEg_val = Eenew/bottom
            return maxEg_val if maxEg_val < grid_Emax else grid_Emax
        else:
            return grid_Emax

###############################
#
#  GRID
#
###############################        

cdef public class Grid [object CGrid, type TGrid ]:
    """grid class"""
    cdef public Egrid
    cdef public xgrid
    cdef public dEdxgrid
    cdef  double Emin
    cdef  double Emax
    cdef  double E0
    cdef  double xmin
    cdef  double xmax
    cdef  int Ngrid
    cdef  double dx
    cdef double *xgrid_data
    cdef double *Egrid_data
    cdef double *dEdxgrid_data
    cdef double di

    def __init__(self, double Emin, double Emax, double E0, int Ngrid, double di = 0.5):
        """ Full constructor: allocates memory and generates the grid """
        self.Ngrid = Ngrid
        self.xgrid = np.zeros((self.Ngrid),dtype=DTYPE)
        self.Egrid = np.zeros((self.Ngrid),dtype=DTYPE)
        self.dEdxgrid = np.zeros((self.Ngrid),dtype=DTYPE)
        self.set_grid( Emin, Emax, E0, di )

    @classmethod
    def fromGrid(cls, Grid grid):
        return cls( grid.Emin, grid.Emax, grid.E0, grid.Ngrid, grid.di )

    @classmethod
    def empty(cls, int Ngrid):
        return cls( 1, 2, 0.5, Ngrid, 0.5 )

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef set_grid(self, double Emin, double Emax, double E0, double di ):
        """ Same as Grid() but without reallocation of memory """
        cdef int i
        cdef int dim = self.Ngrid
        self.Emin = Emin
        self.Emax = Emax
        self.E0 = E0
        self.di = di
        self.xmax = log( self.Emax-self.E0 )
        self.xmin = log( self.Emin-self.E0 )
        self.dx = (self.xmax - self.xmin) * 1.0 / dim
        #get direct C pointers to numpy arrays' data fields
        self.xgrid_data = get_data(self.xgrid)
        self.Egrid_data = get_data(self.Egrid)
        self.dEdxgrid_data = get_data(self.dEdxgrid)
        for i from 0 <= i < dim:
            self.xgrid_data[i] = self.xmin + self.dx*(i+self.di)
            self.Egrid_data[i] = self.E0 + exp( self.xgrid_data[i] )
            self.dEdxgrid_data[i] = self.Egrid_data[i] - self.E0

    cpdef double get_dx(self):
        return self.dx

    cpdef double get_di(self):
        return self.di

    cpdef double get_Emin(self):
        return self.Emin

    cpdef double get_Emax(self):
        return self.Emax

    cpdef double get_E0(self):
        return self.E0

    cpdef double get_Ngrid(self):
        return self.Ngrid

    cpdef double set_di(self, double di):
        cdef double olddi = self.di
        self.set_grid( self.Emin, self.Emax, self.E0, di )
        return olddi

    cdef int iofx(self, double xval) nogil:
        """ Returns the index of the cell containing xval """
        cdef int ival
        ival = int( (xval-self.xmin)/self.dx - self.di )
        return ival

    cdef double xofE(self, double Eval) nogil:
        """ Returns the value of x corresponding to Eval """
        cdef double xval
        xval = log(Eval - self.E0)
        return xval

    cdef inline int iofE(self, double Eval) nogil:
        """ Returns the index of the cell containing Eval """
        return int( (log(Eval-self.E0)-self.xmin)/self.dx - self.di )


###############################
#
#  FUNCTION
#
###############################        

cdef public class Func(Grid)  [object CFunc, type TFunc ]:
    """ Function class derived from Grid class """
    
    cdef public func_vec
    cdef double *func_vec_data
    cdef public lfunc_vec
    cdef double *lfunc_vec_data

    def __init__(self, double Emin, double Emax, double E0, int Ngrid, double di = 0.5, func_vec = None):
        Grid.__init__(self, Emin, Emax, E0, Ngrid, di)
        if func_vec is None:
            self.func_vec = np.zeros((self.Ngrid),dtype=DTYPE)+tiny
            self.func_vec_data = get_data(self.func_vec)
            self.lfunc_vec = np.log(self.func_vec)
            self.lfunc_vec_data = get_data(self.lfunc_vec)
        else:
            self.func_vec = np.copy(func_vec)
            self.func_vec_data = get_data(self.func_vec)
            self.lfunc_vec = np.log(self.func_vec)
            self.lfunc_vec_data = get_data(self.lfunc_vec)

    cpdef set_grid(self, double Emin, double Emax, double E0, double di):
        """ Same as Grid() but without reallocation of memory """
        Grid.set_grid( self, Emin, Emax, E0, di )

    @classmethod
    def fromGrid(cls, Grid grid):
        return cls( grid.Emin, grid.Emax, grid.E0, grid.Ngrid, di = grid.di )

    @classmethod
    def fromFunc(cls, Func f):
        return cls( f.Emin, f.Emax, f.E0, f.Ngrid, di = f.di, func_vec = f.func_vec )

    @classmethod
    def empty(cls, int Ngrid):
        return cls( 1, 2, 0.5, Ngrid, 0.5 )

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef np.ndarray[double, ndim=1] fofE_vec(self, np.ndarray[double, ndim=1] Eval):
        cdef double *Eval_data = get_data(Eval)
        cdef Einterp = np.zeros_like(Eval)
        cdef double *Einterp_data = get_data(Einterp)
        cdef int len = Eval.shape[0]
        cdef int i
        for i from 0 <= i < len:
            Einterp_data[i] = self.fofE(Eval_data[i])
        return Einterp

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef double norm(self) nogil:
        cdef int i
        cdef double norm = 0
        for i from 0 <= i < self.Ngrid:
            norm += self.func_vec_data[i]*self.dEdxgrid_data[i]
        norm *= self.dx
        return norm

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef double Etot(self) nogil:
        cdef int i
        cdef double norm = 0
        for i from 0 <= i < self.Ngrid:
            norm += self.Egrid_data[i] * self.func_vec_data[i]*self.dEdxgrid_data[i]
        norm *= self.dx
        return norm

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef double fofE(self, double Eval) nogil:
        """ Linearly interpolates f(E) in log-log """
        cdef int i
        cdef double logfl, logfr, logf, f, invldiff
        cdef double x, dx
        if Eval < self.Egrid_data[0] or Eval > self.Egrid_data[self.Ngrid-1]:
            return 0
        i = int( (log(Eval-self.E0)-self.xmin)/self.dx - self.di )
        #i = self.iofE(Eval)
        #i = Grid.iofE( self, Eval )
        if i < 0 or i >= self.Ngrid-1:
            return 0
        #log-log
        x  = log(Eval-self.E0)
        dx = (x-self.xgrid_data[i])/self.dx
        logfl = self.lfunc_vec_data[i]
        logfr = self.lfunc_vec_data[i+1]
        return exp(logfr * dx + logfl * (1-dx))
        
    def set_func(self, func_vec):
        return self.set_func_c( get_data(func_vec) )

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef double set_func_c(self, double *func_vec_data) nogil:
        cdef int i
        for i from 0 <= i < self.Ngrid:
            self.func_vec_data[i] = func_vec_data[i] #if func_vec_data[i] > tiny else tiny
            self.lfunc_vec_data[i] = log(func_vec_data[i]+tiny)
        return tiny

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef int set_funci_c(self, int i, double f) nogil:
        self.func_vec_data[i] = max(f,tiny)
        self.lfunc_vec_data[i] = log(self.func_vec_data[i])
        return 0

