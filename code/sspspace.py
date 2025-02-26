
import numpy as np
from scipy.stats import qmc
from scipy.stats import special_ortho_group
from scipy.optimize import minimize

import warnings

class SSPSpace:
    def __init__(self, domain_dim: int, ssp_dim: int, axis_matrix=None, phase_matrix=None,
                 domain_bounds=None, length_scale=1):
        '''
        Represents a domain using spatial semantic pointers.
        Parameters:
        -----------
        domain_dim : int 
            The dimensionality of the domain being represented.
        ssp_dim : int
            The dimensionality of the spatial semantic pointer vector.
        axis_matrix : np.ndarray
            A ssp_dim X domain_dim ndarray representing the axis vectors for
            the domain.
        phase_matrix : np.ndarray
            A ssp_dim x domain_dim ndarray representing the frequency 
            components of the SSP representation.
        domain_bounds : np.ndarray
            A domain_dim X 2 ndarray giving the lower and upper bounds of the 
            domain, used in decoding from an ssp to the point it represents.
        length_scale : float or np.ndarray
            Scales values before encoding.
            
        '''
        self.domain_dim = domain_dim
        self.ssp_dim = ssp_dim
        self.length_scale = length_scale * np.ones((self.domain_dim,1))
        
        if domain_bounds is not None:
            assert domain_bounds.shape[0] == domain_dim
        
        self.domain_bounds = domain_bounds
        
        if (axis_matrix is None) & (phase_matrix is None):
            raise RuntimeError("SSP spaces must be defined by either a axis matrix or phase matrix. Use subclasses to construct spaces with predefined axes.")
        elif (phase_matrix is None):
            assert axis_matrix.shape[0] == ssp_dim, f'Expected ssp_dim {axis_matrix.shape[0]}, got {ssp_dim}.'
            assert axis_matrix.shape[1] == domain_dim
            self.axis_matrix = axis_matrix
            self.phase_matrix = (-1.j*np.log(np.fft.fft(axis_matrix,axis=0))).real
        elif (axis_matrix is None):
            assert phase_matrix.shape[0] == ssp_dim
            assert phase_matrix.shape[1] == domain_dim
            self.phase_matrix = phase_matrix
            self.axis_matrix = np.fft.ifft(np.exp(1.j*phase_matrix), axis=0).real
            
    def update_lengthscale(self, scale):
        '''
        Changes the lengthscale being used in the encoding.
        '''
        if not isinstance(scale, np.ndarray) or scale.size == 1:
            self.length_scale = scale * np.ones((self.domain_dim,))
        else:
            assert scale.size == self.domain_dim
            self.length_scale = scale
        assert self.length_scale.size == self.domain_dim
        ### end if
        
    def optimize_lengthscale(self, init_xs, init_ys):
        ls_0 = self.length_scale
        self.length_scale = np.ones((self.domain_dim,1))
        
        def min_func(length_scale):
            init_phis = self.encode(init_xs/ length_scale)
            W = np.linalg.pinv(init_phis.T) @ init_ys
            mu = np.dot(init_phis.T,W)
            diff = init_ys - mu.T
            err = np.sum(np.power(diff, 2))
            return err

        retval = minimize(min_func, x0=ls_0, method='L-BFGS-B', bounds = self.domain_dim*[(1e-8,1e5)])
        self.length_scale = retval.x.reshape(-1,1)
    
    def encode(self,x):
        '''
        Transforms input data into an SSP representation.
        Parameters:
        -----------
        x : np.ndarray
            A (num_samples, domain_dim) array representing data to be encoded.
        Returns:
        --------
        data : np.ndarray
            A (num_samples, ssp_dim) array of the ssp representation of the data
            
        '''
        x = np.atleast_2d(x)

        ls_mat = np.atleast_2d(np.diag(1/self.length_scale.flatten()))
        #print(ls_mat)
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ scaled_x.T), axis=0 ).real
        return data.T
    
    def encode_and_deriv(self,x):
        '''
        Returns the ssp representation of the data and the derivative of
        the encoding.
        Parameters:
        -----------
        x : np.ndarray
            A (num_samples, domain_dim) array representing data to be encoded.
        Returns:
        --------
        data : np.ndarray
            A (num_samples, ssp_dim) array of the ssp representation of the 
            data
        grad : np.ndarray
            A (num_samples, ssp_dim, domain_dim) array of the ssp representation of the data
        '''
        assert x.ndim == 2, f'Expected 2d data (samples, features), got {x.ndim}d data.'
        assert x.shape[1] == self.phase_matrix.shape[1], (f'Expected data to have ' 
                                                         f'{self.phase_matrix.shape[1]} '
                                                         f'features, got {x.shape[1]}.')

#         x= x.reshape(self.domain_dim, -1)
        ls_mat = np.atleast_2d(np.diag(1 / self.length_scale))
        scaled_x = x @ ls_mat
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ scaled_x.T ), axis=0 ).real
        ddata = np.fft.ifft( 1.j * (self.phase_matrix @ ls_mat) @ np.exp( 1.j * self.phase_matrix @ scaled_x.T ), axis=0 ).real
        return data.T, ddata.T
    
    def encode_fourier(self,x):
        assert x.ndim == 2, f'Expected 2d data (samples, features), got {x.ndim} data.'
        assert x.shape[1] == self.phase_matrix.shape[1], (f'Expected data to have ' 
                                                         f'{self.phase_matrix.shape[1]} '
                                                         f'features, got {x.shape[1]}.')
#         x= x.reshape(self.domain_dim, -1)
        data =  np.exp( 1.j * self.phase_matrix @ (x / self.length_scale).T )
        return data.T
    
    #def encode_as_SSP(self,x):
    #    assert x.shape[0] == self.domain_dim
    #    data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ x / self.length_scale ), axis=0 ).real
    #    return SSP(data,self)
    
 
    def decode(self,ssp,method='from-set',sampling_method='grid',
               num_samples =300, samples=None): # other args for specfic methods
        '''
        Transforms ssp representation back into domain representation.
        Parameters:
        -----------
        ssp : np.ndarray
            SSP representation of a data point.
        method : {'from-set', 'direct-optim'}
            The technique for decoding the ssp.  from-set samples the domain
            and finds the closest match under the dot product. direct-optim
            does an initial coarse sampling and then optimizes the decoded
            value starting from the initial best match in the coarse sampling.
        sampling_method : {'grid'|'length-scale'|'sobol'}
            Evenly distributes samples along the domain axes
        num_samples : int
            The number of samples along each axis.
        Returns:
        --------
        x : np.ndarray
            The decoded point
        '''
        if (method=='direct-optim') | (method=='from-set'):
            if samples is None:
                sample_ssps, sample_points = self.get_sample_pts_and_ssps(method=sampling_method, 
                        num_points_per_dim=num_samples)
            else:
                sample_ssps, sample_points = samples
                assert sample_ssps.shape[1] == ssp.shape[1], f'Expected {sample_ssps.shape} dim, got {ssp.shape}'
            

#         unit_ssp = ssp / np.linalg.norm(ssp, axis=1)
        unit_ssp = np.zeros(ssp.shape)
        for s_idx, s in enumerate(ssp):
            unit_ssp[s_idx,:] = s / np.linalg.norm(s)
        
        if method=='from-set': 
            sims = sample_ssps @ unit_ssp.T
            return sample_points[np.argmax(sims),:]
        elif method=='direct-optim':
            def min_func(x,target):
                x_ssp = self.encode(np.atleast_2d(x))
                return -np.inner(x_ssp, target).flatten()

            retvals = np.zeros((ssp.shape[0],self.domain_dim))
            for s_idx, u_ssp in enumerate(unit_ssp):
                x0 = self.decode(np.atleast_2d(u_ssp),
                                 method='from-set',
                                 sampling_method='length-scale', 
                                 num_samples=num_samples, samples=samples)


                soln = minimize(min_func, x0,
                                args=(np.atleast_2d(u_ssp),),
                                method='L-BFGS-B',
                                bounds=self.domain_bounds)
                retvals[s_idx,:] = soln.x
            return retvals #soln.x
        elif method=='network':
            if self.decoder_model is None:
                raise Exception('Network not trained for decoding. You must first call train_decoder_net')
            return self.decoder_model.predict(ssp)
        elif method=='network-optim':
            if self.decoder_model is None:
                raise Exception('Network not trained for decoding. You must first call train_decoder_net')
            x0 = self.decoder_model.predict(ssp)
            

            solns = np.zeros(x0.shape)
            for i in range(x0.shape[0]):
                def min_func(x,target=ssp[i,:]):
                    x_ssp = self.encode(np.atleast_2d(x))
                    return -np.inner(x_ssp, target).flatten()
                soln = minimize(min_func, x0[i,:], 
                            method='L-BFGS-B',
                            bounds=self.domain_bounds)
                solns[i,:] = soln.x
            return solns
        else:
            raise NotImplementedError(f'Unrecognized decoding method: {method}')
        
    def clean_up(self,ssp,method='from-set'):
        if method=='least-squares':
            x = self.decode(ssp,method)
            return self.encode(x)
        elif method=='from-set':
            sample_ssps = self.get_sample_ssps(500)
            sims = sample_ssps.T @ ssp
            return sample_ssps[:,np.argmax(sims)]
        else:
            raise NotImplementedError()
        
    def get_sample_points(self, samples_per_dim=100, method='length-scale', seed=None):
        '''
        Identifies points in the domain of the SSP encoding that 
        will be used to determine optimal decoding.
        Parameters
        ----------
        method: {'grid'|'length-scale'|'sobol'}
            The way to select samples from the domain. 
            'grid' uniformly spaces samples_per_dim points on the domain
            'sobol' decodes using samples_per_dim**data_dim points generated 
                using a sobol sampling
            'length-scale' uses the selected lengthscale to determine the number
                of sample points generated per dimension.
        Returns
        -------
        sample_pts : np.ndarray 
            A (num_samples, domain_dim) array of candidate decoding points.
        '''

        if self.domain_bounds is None:
            bounds = np.vstack([-10*np.ones(self.domain_dim), 10*np.ones(self.domain_dim)]).T
        else:
            bounds = self.domain_bounds

        if method == 'grid':
            num_pts_per_dim = [samples_per_dim for _ in range(bounds.shape[0])]
        elif method == 'length-scale':
            num_pts_per_dim = [2*int(np.ceil((b[1]-b[0])/self.length_scale[b_idx])) for b_idx, b in enumerate(bounds)]
        else:
            num_pts_per_dim = samples_per_dim 

        if method=='grid' or method=='length-scale':
            xxs = np.meshgrid(*[np.linspace(bounds[i,0], 
                                            bounds[i,1],
                                            num_pts_per_dim[i]
                                        ) for i in range(self.domain_dim)])
            retval = np.array([x.reshape(-1) for x in xxs]).T
            assert retval.shape[1] == self.domain_dim, f'Expected {self.domain_dim}d data, got {retval.shape[1]}d data'
            return retval

        elif method=='sobol':
            ## check bounds are within range to prevent overflow
            if np.all(bounds[:,0] > -1e16) and np.all(bounds[:,1] < 1e16):
                lbounds = bounds[:,0]
                ubounds = bounds[:,1]
            else:
                lbounds = bounds[:,0].astype(np.float64)
                ubounds = bounds[:,1].astype(np.float64)
            
            num_points = np.prod(num_pts_per_dim)
            sampler = qmc.Sobol(d=self.domain_dim, seed=seed) 
            u_sample_points = sampler.random(num_points)
            sample_points = qmc.scale(u_sample_points, lbounds, ubounds).T
                
        elif method=='Rd':
            num_points = np.prod(samples_per_dim)
            u_sample_points = _Rd_sampling(num_points, self.domain_dim)
            lbounds = bounds[:,0]
            ubounds = bounds[:,1]
            sample_points = qmc.scale(u_sample_points, lbounds, ubounds).T
        else:
            raise NotImplementedError(f'Sampling method {method} is not implemented')
        return sample_points.T 
        
    
    def get_sample_ssps(self,num_points): 
        sample_points = self.get_sample_points(num_points)
        sample_ssps = self.encode(sample_points)
        return sample_ssps
    
    def get_sample_pts_and_ssps(self,num_points_per_dim=100, method='grid'): 
        sample_points = self.get_sample_points(
                                        method=method, 
                                        samples_per_dim=num_points_per_dim
                                    )
        if method == 'grid':
            expected_points = int(num_points_per_dim**(self.domain_dim))
            assert sample_points.shape[0] == expected_points, f'Expected {expected_points} samples, got {sample_points.shape[0]}.'

        sample_ssps = self.encode(sample_points)

        if method == 'grid':
            assert sample_ssps.shape[0] == expected_points

        return sample_ssps, sample_points
    
    def normalize(self,ssp):
        return ssp/np.sqrt(np.sum(ssp**2))
    
    def make_unitary(self,ssp):
        fssp = np.fft.fft(ssp)
        fssp = fssp/np.sqrt(fssp.real**2 + fssp.imag**2)
        return np.fft.ifft(fssp).real  
    
    def identity(self):
        s = np.zeros(self.ssp_dim)
        s[0] = 1
        return s
    
    def bind(self,a,b):
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        return np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b,axis=1), axis=1).real
    
    def invert(self,a):
        a = np.atleast_2d(a)
        return a[:,-np.arange(self.ssp_dim)]
    
    def similarity_plot(self,ssp,n_grid=100,plot_type='heatmap',ax=None,**kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        if self.domain_dim == 1:
            xs = np.linspace(self.domain_bounds[0,0],self.domain_bounds[0,1], n_grid)
            sims = ssp @ self.encode(np.atleast_2d(xs).T).T
            im = ax.plot(xs, sims.reshape(-1) )
            ax.set_xlim(self.domain_bounds[0,0],self.domain_bounds[0,1])
        elif self.domain_dim == 2:
            xs = np.linspace(self.domain_bounds[0,0],self.domain_bounds[0,1], n_grid)
            ys = np.linspace(self.domain_bounds[1,0],self.domain_bounds[1,1], n_grid)
            X,Y = np.meshgrid(xs,ys)
            sims = ssp @ self.encode(np.vstack([X.reshape(-1),Y.reshape(-1)]).T).T 
            if plot_type=='heatmap':
                im = ax.pcolormesh(X,Y,sims.reshape(X.shape),**kwargs)
            elif plot_type=='contour':
                im = ax.contour(X,Y,sims.reshape(X.shape),**kwargs)
            elif plot_type=='contourf':
                im = ax.contourf(X,Y,sims.reshape(X.shape),**kwargs)
            ax.set_xlim(self.domain_bounds[0,0],self.domain_bounds[0,1])
            ax.set_ylim(self.domain_bounds[1,0],self.domain_bounds[1,1])
        else:
            raise NotImplementedError()
        return im
    
    def train_decoder_net(self,n_training_pts=200000,n_hidden_units = 8,
                          learning_rate=1e-3,n_epochs = 20, load_file=True, save_file=True):
        import tensorflow as tf
        tf.config.set_visible_devices([],'GPU')

        import sklearn
        from tensorflow import keras
        from keras import layers, regularizers
        
        if (type(self).__name__ == 'HexagonalSSPSpace'):
            path_name = './saved_decoder_nets/domaindim' + str(self.domain_dim) + '_lenscale' + str(self.length_scale[0]) + '_nscales' + str(self.n_scales) + '_nrotates' + str(self.n_rotates) + '_scale_min' + str(self.scale_min) + '_scalemax' + str(self.scale_max) +'.h5'
        else:
            #warnings.warn("Cannot load decoder net for non HexagonalSSPSpace class")
            load_file = False
            save_file=False
        
        
        if load_file:
            try:
                self.decoder_model = keras.models.load_model(path_name)
                return
            except BaseException as be:
                print('Error loading decoder:')
                print(be)
                pass

        model = keras.Sequential([
             layers.Dense(self.ssp_dim, activation="relu", name="layer1"),# layers.Dropout(.1),
             layers.Dense(n_hidden_units, activation="relu", name="layer2"), # kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
             layers.Dense(self.domain_dim, name="output"),
            ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error')
        
        sample_ssps, sample_points = self.get_sample_pts_and_ssps(num_points_per_dim=n_training_pts,
                                                                           method='Rd')
        shuffled_ssps, shuffled_pts = sklearn.utils.shuffle(sample_ssps, sample_points)
        history = model.fit(shuffled_ssps, shuffled_pts,
            epochs=n_epochs,verbose=True, validation_split = 0.1)
        
        if save_file:
           model.save(path_name)

            
        self.decoder_model = model
        return history
            
class RandomSSPSpace(SSPSpace):
    '''
    Creates an SSP space using randomly generated frequency components.
    '''
    def __init__(self, domain_dim: int, ssp_dim: int,  domain_bounds=None, length_scale=1, rng=np.random.default_rng()):
#         partial_phases = rng.random.rand(ssp_dim//2,domain_dim)*2*np.pi - np.pi
        
        
        #partial_phases = rng.random((ssp_dim // 2, domain_dim)) * 2 * np.pi - np.pi
        #axis_matrix = _constructaxisfromphases(partial_phases)
        def make_good_unitary(dim, eps=1e-3, rng=np.random):
            a = rng.rand((dim - 1) // 2)
            sign = rng.choice((-1, +1), len(a))
            phi = sign * np.pi * (eps + a * (1 - 2 * eps))
            assert np.all(np.abs(phi) >= np.pi * eps)
            assert np.all(np.abs(phi) <= np.pi * (1 - eps))
        
            fv = np.zeros(dim, dtype='complex64')
            fv[0] = 1
            fv[1:(dim + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
            fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
            if dim % 2 == 0:
                fv[dim // 2] = 1
        
            assert np.allclose(np.abs(fv), 1)
            v = np.fft.ifft(fv)
            
            v = v.real
            assert np.allclose(np.fft.fft(v), fv)
            assert np.allclose(np.linalg.norm(v), 1)
            return v

        axis_matrix = np.zeros((ssp_dim,domain_dim))
        for i in range(domain_dim):
            axis_matrix[:,i] = make_good_unitary(ssp_dim)

        super().__init__(domain_dim, 
                         axis_matrix.shape[0],
                         axis_matrix=axis_matrix,
                         domain_bounds=domain_bounds,
                         length_scale=length_scale,
                         )
### end class RandomSSPSpace ###

class GaussianSSPSpace(SSPSpace):
    '''
    Creates an SSP space using randomly generated frequency components.
    '''
    def __init__(self, 
                 domain_dim: int,
                 ssp_dim: int,
                 domain_bounds=None,
                 length_scale=1,
                 rng=np.random.default_rng()):

        partial_phases = rng.normal(loc=0,
                                    scale=1,
                                    size=(ssp_dim//2, domain_dim)
                                    )#*2*np.pi - np.pi
        #partial_phases = rng.random((ssp_dim // 2, domain_dim)) * 2 * np.pi - np.pi
        axis_matrix = _constructaxisfromphases(partial_phases)
#         ### TODO: should the rng default argument for make_good_unitary be the 
#         # rng passed in through the default __init__ argument?
#         def make_good_unitary(dim, eps=1e-3, rng=np.random):
#             a = rng.normal(loc=0, scale=np.pi, size=dim)#size=(dim - 1) // 2)
#             phi = a
# #             phi = (eps + a * (1 - 2 * eps))
# #             sign = rng.choice((-1, +1), len(a))
# #             phi = sign * (eps + a * (1 - 2 * eps))
# #             assert np.all(np.abs(phi) >= eps)
# #             assert np.all(np.abs(phi) <= (1 - eps))
#         
#             fv = np.zeros(dim, dtype='complex64')
#             fv = np.cos(phi) + 1j * np.sin(phi)
# #             fv[0] = 1
# #             fv[1:(dim + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
# #             fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
#             if dim % 2 == 0:
#                 fv[dim // 2] = 1
#             else:
#                 fv[0] = 1
#         
# #             assert np.allclose(np.abs(fv), 1)
#             v = np.fft.ifft(fv)
#             
#             v = v.real
# #             assert np.allclose(np.fft.fft(v), fv)
# #             assert np.allclose(np.linalg.norm(v), 1)
#             return v
# 
#         axis_matrix = np.zeros((ssp_dim,domain_dim))
#         for i in range(domain_dim):
#             axis_matrix[:,i] = make_good_unitary(ssp_dim)
# 
        super().__init__(domain_dim,
                         axis_matrix.shape[0],
                         axis_matrix=axis_matrix,
                         domain_bounds=domain_bounds,
                         length_scale=length_scale,
                         )
### end class GaussianSSPSpace ###


        
class HexagonalSSPSpace(SSPSpace):
    '''
    Creates an SSP space using the Hexagonal Tiling developed by NS Dumont 
    (2020)
    '''
    def __init__(self,  domain_dim:int,ssp_dim: int=151, n_rotates:int=5, n_scales:int=5, 
                 scale_min=0.1, scale_max=3,
                 domain_bounds=None, length_scale=1):
        if (n_rotates==5) & (n_scales==5) & (ssp_dim!=151): # user wants to define ssp with total dim, not number of simplex rotates and scales
            n_rotates = int(np.sqrt((ssp_dim-1)/(2*(domain_dim+1))))
            n_scales = n_rotates
            ssp_dim = n_rotates*n_scales*(domain_dim+1)*2 + 1
            
        phases_hex = np.hstack([np.sqrt(1+ 1/domain_dim)*np.identity(domain_dim) - (domain_dim**(-3/2))*(np.sqrt(domain_dim+1) + 1),
                         (domain_dim**(-1/2))*np.ones((domain_dim,1))]).T
        
        self.grid_basis_dim = domain_dim + 1
        self.num_grids = n_rotates*n_scales
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.n_scales = n_scales
        self.n_rotates = n_rotates
        
        scales = np.linspace(scale_min,scale_max,n_scales)
        phases_scaled = np.vstack([phases_hex*i for i in scales])
        
        if (n_rotates==1):
            phases_scaled_rotated = phases_scaled
        elif (domain_dim==1):
            scales = np.linspace(scale_min,scale_max,n_scales+n_rotates)
            phases_scaled_rotated = np.vstack([phases_hex*i for i in scales])
        elif (domain_dim == 2):
            angles = np.linspace(0,2*np.pi/3,n_rotates,endpoint=False)
            R_mats = np.stack([np.stack([np.cos(angles), -np.sin(angles)],axis=1),
                        np.stack([np.sin(angles), np.cos(angles)], axis=1)], axis=1)
            phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0,2,1).reshape(-1,domain_dim)
        else:
            R_mats = special_ortho_group.rvs(domain_dim, size=n_rotates, random_state=1)
            phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0,2,1).reshape(-1,domain_dim)
        
        axis_matrix = _constructaxisfromphases(phases_scaled_rotated)
        ssp_dim = axis_matrix.shape[0]
        super().__init__(domain_dim,ssp_dim,axis_matrix=axis_matrix,
                       domain_bounds=domain_bounds,length_scale=length_scale)
  
    # def sample_grid_encoders_old(self, n):
        # sample_pts = self.get_sample_points(n)
        # N = self.num_grids
        # if N < n:
            # sorts = np.hstack([np.arange(N), np.random.randint(0, N - 1, size = n - N)])
        # else:
            # sorts = np.arange(n)
        # encoders = np.zeros((self.ssp_dim,n))
        # for i in range(n):
            # sub_mat = _get_sub_SSP(sorts[i],N,sublen=self.grid_basis_dim)
            # proj_mat = _proj_sub_SSP(sorts[i],N,sublen=self.grid_basis_dim)
            # sub_space = SSPSpace(self.domain_dim,2*self.grid_basis_dim + 1, axis_matrix= sub_mat @ self.axis_matrix)
            # encoders[:,i] = N * proj_mat @ sub_space.encode(sample_pts[:,i])
        # return encoders
        
    def sample_grid_encoders(self, n, seed=None):
        sample_pts = self.get_sample_points(n, method='sobol', seed=seed)
        N = self.num_grids
        if N < n:
            np.random.seed(seed)
            sorts = np.hstack([np.arange(N), np.random.randint(0, N - 1, size = n - N)])
        else:
            sorts = np.arange(n)
        encoders = np.zeros((n,self.ssp_dim))
        for i in range(n):
            sub_mat = _get_sub_SSP(sorts[i],N,sublen=self.grid_basis_dim)
            proj_mat = _proj_sub_SSP(sorts[i],N,sublen=self.grid_basis_dim)
            sub_space = SSPSpace(self.domain_dim,2*self.grid_basis_dim + 1, axis_matrix= sub_mat @ self.axis_matrix)
            encoders[i,:] = (N * proj_mat @ sub_space.encode(np.atleast_2d(sample_pts[i,:])).T).reshape(-1)
        return encoders

class RdSSPSpace(SSPSpace):
    def __init__(self,  domain_dim:int,ssp_dim: int=151,  
                 scale_max=1,
                 domain_bounds=None, length_scale=1): 
        # From nengolib
        from scipy.special import beta, betainc, betaincinv
        from nengo.dists import UniformHypersphere
        from scipy.linalg import svd
        
        def _rd_generate(n, d, seed=0.5):
            """Generates the first ``n`` points in the ``R_d`` sequence."""
        
            # http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
            def gamma(d, n_iter=20):
                """Newton-Raphson-Method to calculate g = phi_d."""
                x = 1.0
                for _ in range(n_iter):
                    x -= (x**(d + 1) - x - 1) / ((d + 1) * x**d - 1)
                return x
        
            g = gamma(d)
            alpha = np.zeros(d)
            for j in range(d):
                alpha[j] = (1/g) ** (j + 1) % 1
        
            z = np.zeros((n, d))
            z[0] = (seed + alpha) % 1
            for i in range(1, n):
                z[i] = (z[i-1] + alpha) % 1
        
            return z
        
        def spherical_transform(samples):
            samples = np.asarray(samples)
            samples = samples[:, None] if samples.ndim == 1 else samples
            coords = np.empty_like(samples)
            n, d = coords.shape
        
            # inverse transform method (section 1.5.2)
            for j in range(d):
                coords[:, j] = (np.pi * np.sin(np.pi * samples[:, j]) ** (d-j-1) / beta((d-j) / 2., .5))
                
            # spherical coordinate transform
            mapped = np.ones((n, d+1))
            i = np.ones(d)
            i[-1] = 2.0
            s = np.sin(i[None, :] * np.pi * coords)
            c = np.cos(i[None, :] * np.pi * coords)
            mapped[:, 1:] = np.cumprod(s, axis=1)
            mapped[:, :-1] *= c
            return mapped
        
        def random_orthogonal(d, rng=None):
            rng = np.random if rng is None else rng
            m = UniformHypersphere(surface=True).sample(d, d, rng=rng)
            u, s, v = svd(m)
            return np.dot(u, v)
        
        n = ssp_dim//2
        if domain_dim == 1:
            samples = np.linspace(1./n, 1., n)[:, None]
        else:
            samples = _rd_generate(n, domain_dim)
        samples, radius = samples[:, :-1], samples[:, -1:] ** (1. / domain_dim)

        mapped = spherical_transform(samples)

        # radius adjustment for ball versus sphere, and a random rotation
        rotation = random_orthogonal(domain_dim)
        phases = np.dot(mapped * radius, rotation)
        
        axis_matrix = _constructaxisfromphases(phases)
        ssp_dim = axis_matrix.shape[0]
        super().__init__(domain_dim,ssp_dim,axis_matrix=axis_matrix,
                       domain_bounds=domain_bounds,length_scale=length_scale)
        
        
        
    
def _constructaxisfromphases(K):
    d = K.shape[0]
    F = np.ones((d*2 + 1,K.shape[1]), dtype="complex")
    F[0:d,:] = np.exp(1.j*K)
    F[-d:,:] = np.flip(np.conj(F[0:d,:]),axis=0)
    axes =  np.fft.ifft(np.fft.ifftshift(F,axes=0),axis=0).real
    return axes

def _get_sub_FourierSSP(n, N, sublen=3):
    # Return a matrix, \bar{A}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \bar{A}_n F{S_{total}} = F{S_n}
    # i.e. pick out the sub vector in the Fourier domain
    tot_len = 2*sublen*N + 1
    FA = np.zeros((2*sublen + 1, tot_len))
    FA[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FA[sublen, sublen*N] = 1
    FA[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FA

def _get_sub_SSP(n,N,sublen=3):
    # Return a matrix, A_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # A_n S_{total} = S_n
    # i.e. pick out the sub vector in the time domain
    tot_len = 2*sublen*N + 1
    FA = _get_sub_FourierSSP(n,N,sublen=sublen)
    W = np.fft.fft(np.eye(tot_len))
    invW = np.fft.ifft(np.eye(2*sublen + 1))
    A = invW @ np.fft.ifftshift(FA) @ W
    return A.real

def _proj_sub_FourierSSP(n,N,sublen=3):
    # Return a matrix, \bar{B}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n \bar{B}_n F{S_{n}} = F{S_{total}}
    # i.e. project the sub vector in the Fourier domain such that summing all such projections gives the full vector in Fourier domain
    tot_len = 2*sublen*N + 1
    FB = np.zeros((2*sublen + 1, tot_len))
    FB[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FB[sublen, sublen*N] = 1/N # all sub vectors have a "1" zero freq term so scale it so full vector will have 1 
    FB[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FB.T

def _proj_sub_SSP(n,N,sublen=3):
    # Return a matrix, B_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n B_n S_{n} = S_{total}
    # i.e. project the sub vector in the time domain such that summing all such projections gives the full vector
    tot_len = 2*sublen*N + 1
    FB = _proj_sub_FourierSSP(n,N,sublen=sublen)
    invW = np.fft.ifft(np.eye(tot_len))
    W = np.fft.fft(np.eye(2*sublen + 1))
    B = invW @ np.fft.ifftshift(FB) @ W
    return B.real

def _Rd_sampling(n,d,seed=0.5):
    def phi(d): 
        x=2.0000 
        for i in range(10): 
          x = pow(1+x,1/(d+1)) 
        return x
    g = phi(d) 
    alpha = np.zeros(d) 
    for j in range(d): 
      alpha[j] = pow(1/g,j+1) %1 
    z = np.zeros((n, d)) 
    for i in range(n):
        z[i] = seed + alpha*(i+1)
    z = z %1
    return 