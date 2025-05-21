# imports from other packages
from __future__ import print_function, division

from numpy import array, ones, full, hanning, hamming, bartlett, blackman, \
invert, dot, newaxis, zeros, empty, fft, float32, float64, complex64, linalg, \
where, searchsorted, pi, multiply, sign, diag, arange, sqrt, exp, log10, int,\
reshape, hstack, vstack, eye, tril, size, clip, tile, round, delete, \
absolute, argsort, sort, sum, hsplit, fill_diagonal, zeros_like, isclose, \
vdot, flatnonzero, einsum, ndarray, isscalar, inf

from sklearn.linear_model import LassoLars, LassoLarsCV, LassoLarsIC,\
OrthogonalMatchingPursuit, ElasticNet, OrthogonalMatchingPursuitCV, Lasso

from scipy.optimize import nnls, linprog, fmin_l_bfgs_b
from scipy.linalg import inv, eigh, eigvals, fractional_matrix_power
from warnings import warn

from traits.api import HasPrivateTraits, Float, Int, ListInt, ListFloat, \
CArray, Property, Instance, Trait, Bool, Range, Delegate, Enum, Any, \
cached_property, on_trait_change, property_depends_on
from traits.trait_errors import TraitError

class PointSpreadFunction (HasPrivateTraits):
    """
    The point spread function.
    
    This class provides tools to calculate the PSF depending on the used 
    microphone geometry, focus grid, flow environment, etc.
    The PSF is needed by several deconvolution algorithms to correct
    the aberrations when using simple delay-and-sum beamforming.
    """
    
    # Instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes
    # that contains information about the steering vector. This is a private trait.
    # Do not set this directly, use `steer` trait instead.
    _steer_obj = Instance(SteeringVector(), SteeringVector)   
    
    #: :class:`~acoular.fbeamform.SteeringVector` or derived object. 
    #: Defaults to :class:`~acoular.fbeamform.SteeringVector` object.
    steer = Property(desc="steering vector object")  
    
    def _get_steer(self):
        return self._steer_obj
    
    def _set_steer(self, steer):
        if isinstance(steer, SteeringVector):
            self._steer_obj = steer
        elif steer in ('true level', 'true location', 'classic', 'inverse'):
            # Type of steering vectors, see also :ref:`Sarradj, 2012<Sarradj2012>`.
            warn("Deprecated use of 'steer' trait. "
                 "Please use object of class 'SteeringVector' in the future.", 
                 Warning, stacklevel = 2)
            self._steer_obj = SteeringVector(steer_type = steer)
        else:
            raise(TraitError(args=self,
                             name='steer', 
                             info='SteeringVector',
                             value=steer))

    # --- List of backwards compatibility traits and their setters/getters -----------
    
    # :class:`~acoular.environments.Environment` or derived object. 
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait.
    env = Property()
    
    def _get_env(self):
        return self._steer_obj.env    
    
    def _set_env(self, env):
        warn("Deprecated use of 'env' trait. ", Warning, stacklevel = 2)
        self._steer_obj.env = env
    
    # The speed of sound.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait.
    c = Property()
    
    def _get_c(self):
        return self._steer_obj.env.c
    
    def _set_c(self, c):
        warn("Deprecated use of 'c' trait. ", Warning, stacklevel = 2)
        self._steer_obj.env.c = c
   
    # :class:`~acoular.grids.Grid`-derived object that provides the grid locations.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait.
    grid = Property()

    def _get_grid(self):
        return self._steer_obj.grid
    
    def _set_grid(self, grid):
        warn("Deprecated use of 'grid' trait. ", Warning, stacklevel = 2)
        self._steer_obj.grid = grid
    
    # :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait
    mpos = Property()
    
    def _get_mpos(self):
        return self._steer_obj.mics
    
    def _set_mpos(self, mpos):
        warn("Deprecated use of 'mpos' trait. ", Warning, stacklevel = 2)
        self._steer_obj.mics = mpos
    
    
    # Sound travel distances from microphone array center to grid points (r0)
    # and all array mics to grid points (rm). Readonly.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait
    r0 = Property()
    def _get_r0(self):
        return self._steer_obj.r0
    
    rm = Property()
    def _get_rm(self):
        return self._steer_obj.rm
    
    # --- End of backwards compatibility traits --------------------------------------
    
    
    #: Indices of grid points to calculate the PSF for.
    grid_indices = CArray( dtype=int, value=array([]), 
                     desc="indices of grid points for psf") #value=array([]), value=self.grid.pos(),
    
    #: Flag that defines how to calculate and store the point spread function
    #: defaults to 'single'.
    #:
    #: * 'full': Calculate the full PSF (for all grid points) in one go (should be used if the PSF at all grid points is needed, as with :class:`DAMAS<BeamformerDamas>`)
    #: * 'single': Calculate the PSF for the grid points defined by :attr:`grid_indices`, one by one (useful if not all PSFs are needed, as with :class:`CLEAN<BeamformerClean>`)
    #: * 'block': Calculate the PSF for the grid points defined by :attr:`grid_indices`, in one go (useful if not all PSFs are needed, as with :class:`CLEAN<BeamformerClean>`)
    #: * 'readonly': Do not attempt to calculate the PSF since it should already be cached (useful if multiple processes have to access the cache file)
    calcmode = Trait('single', 'block', 'full', 'readonly',
                     desc="mode of calculation / storage")
              
    #: Floating point precision of property psf. Corresponding to numpy dtypes. Default = 64 Bit.
    precision = Trait('float64', 'float32',
            desc="precision (32/64 Bit) of result, corresponding to numpy dtypes")
    
    #: The actual point spread function.
    psf = Property(desc="point spread function")
    
    #: Frequency to evaluate the PSF for; defaults to 1.0. 
    freq = Float(1.0, desc="frequency")

    # hdf5 cache file
    h5f = Instance( H5CacheFileBase, transient = True )
    
    # internal identifier
    digest = Property( depends_on = ['_steer_obj.digest', 'precision'], cached = True)

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    def _get_filecache( self ):
        """
        function collects cached results from file depending on 
        global/local caching behaviour. Returns (None, None) if no cachefile/data 
        exist and global caching mode is 'readonly'.
        """
        filename = 'psf' + self.digest
        nodename = ('Hz_%.2f' % self.freq).replace('.', '_')
#        print("get cachefile:", filename)
        H5cache.get_cache_file( self, filename ) 
        if not self.h5f: # only happens in case of global caching readonly
#            print("no cachefile:", filename)
            return (None, None)# only happens in case of global caching readonly
                    
        if config.global_caching == 'overwrite' and self.h5f.is_cached(nodename):
#            print("remove existing data for nodename",nodename)
            self.h5f.remove_data(nodename) # remove old data before writing in overwrite mode
        
        if not self.h5f.is_cached(nodename):
#            print("no data existent for nodename:", nodename)
            if config.global_caching == 'readonly':
                return (None, None)
            else:
#                print("initialize data.")
                gs = self.steer.grid.size
                group = self.h5f.create_new_group(nodename)
                self.h5f.create_compressible_array('result',
                                      (gs, gs),
                                      self.precision,
                                      group)
                self.h5f.create_compressible_array('gridpts',
                                      (gs,),
                                      'int8',#'bool', 
                                      group)
        ac = self.h5f.get_data_by_reference('result','/'+nodename)
        gp = self.h5f.get_data_by_reference('gridpts','/'+nodename)
        return (ac,gp)        

    def _get_psf ( self ):
        """
        This is the :attr:`psf` getter routine.
        The point spread function is either loaded or calculated.
        """
        gs = self.steer.grid.size
        if not self.grid_indices.size: 
            self.grid_indices = arange(gs)

        if not config.global_caching == 'none':
#            print("get filecache..")
            (ac,gp) = self._get_filecache()
            if ac and gp: 
#                print("cached data existent")
                if not gp[:][self.grid_indices].all():
#                    print("calculate missing results")                            
                    if self.calcmode == 'readonly':
                        raise ValueError('Cannot calculate missing PSF (points) in \'readonly\' mode.')
                    if config.global_caching == 'readonly':
                        (ac, gp) = (ac[:], gp[:])
                        self.calc_psf(ac,gp)
                        return ac[:,self.grid_indices]
                    else:
                        self.calc_psf(ac,gp)
                        self.h5f.flush()
                        return ac[:,self.grid_indices]
#                else:
#                    print("cached results are complete! return.")
                return ac[:,self.grid_indices]
            else: # no cached data/file
#                print("no caching, calculate result")
                ac = zeros((gs, gs), dtype=self.precision)
                gp = zeros((gs,), dtype='int8')
                self.calc_psf(ac,gp)
        else: # no caching activated
#            print("no caching activated, calculate result")
            ac = zeros((gs, gs), dtype=self.precision)
            gp = zeros((gs,), dtype='int8')
            self.calc_psf(ac,gp)
        return ac[:,self.grid_indices] 

    def calc_psf( self, ac, gp ):
        """
        point-spread function calculation
        """
        if self.calcmode != 'full':
            # calc_ind has the form [True, True, False, True], except
            # when it has only 1 entry (value True/1 would be ambiguous)
            if self.grid_indices.size == 1:
                calc_ind = [0]
            else:
                calc_ind = invert(gp[:][self.grid_indices])
        
        # get indices which have the value True = not yet calculated
            g_ind_calc = self.grid_indices[calc_ind]
        
        if self.calcmode == 'single': # calculate selected psfs one-by-one
            for ind in g_ind_calc:
                ac[:,ind] = self._psfCall([ind])[:,0]
                gp[ind] = 1
        elif self.calcmode == 'full': # calculate all psfs in one go
            gp[:] = 1
            ac[:] = self._psfCall(arange(self.steer.grid.size))
        else: # 'block' # calculate selected psfs in one go
            hh = self._psfCall(g_ind_calc)
            indh = 0
            for ind in g_ind_calc:
                gp[ind] = 1
                ac[:,ind] = hh[:,indh]
                indh += 1

    def _psfCall(self, ind):
        """
        Manages the calling of the core psf functionality.
        
        Parameters
        ----------
        ind : list of int
            Indices of gridpoints which are assumed to be sources.
            Normalization factor for the beamforming result (e.g. removal of diag is compensated with this.)

        Returns
        -------
        The psf [1, nGridPoints, len(ind)]
        """
        if type(self.steer) == SteeringVector: # for simple steering vector, use faster method
            result = calcPointSpreadFunction(self.steer.steer_type, 
                                             self.steer.r0, 
                                             self.steer.rm, 
                                             2*pi*self.freq/self.env.c, 
                                             ind, self.precision)
        else: # for arbitrary steering sectors, use general calculation
            # there is a version of this in fastFuncs, may be used later after runtime testing and debugging
            product = dot(self.steer.steer_vector(self.freq).conj(), self.steer.transfer(self.freq,ind).T)
            result = (product * product.conj()).real
        return result