import numpy as np
import nengo
import pytry
import scipy
import nni

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from datetime import datetime
import sys,os

sys.path.insert(0,'.')
import sspspace

# Define system dynamics
class pen:
    def __init__(self,dt,m,g,l,b,t_dist):
        self.dt = dt
        self.g = g
        self.l = l
        self.b = b
        self.m = m
        self.t_dist = t_dist
    def __call__(self,t, x):
        if t > self.t_dist:
            dis = x[0] **2
        else:
            dis = 0.

        f_x = np.asarray([ x[1] + dis, -self.m*self.g*self.l * np.sin(x[0]) - self.b*x[1] + 10*x[2]])
        
        return x[0:2] + f_x * self.dt

# helper functions
def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))

def make_unitary_matrix_fourier( ssp_dim, domain_dim, eps=1e-3, rng = np.random, psd_sampling = 'uniform' ):
    if psd_sampling == 'gaussian':
        # gaussian kernel
        a = rng.normal( loc = 0., scale = 1., size = ( (ssp_dim - 1)//2, domain_dim) )
        phi = np.pi * (eps + a * (1 - 2 * eps))
    
    elif psd_sampling == 'uniform':
        # sinc kernel
        a = rng.rand( (ssp_dim - 1)//2, domain_dim )
        sign = rng.choice((-1, +1), size=np.shape(a) )
        phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    
    fv = np.zeros( (ssp_dim,domain_dim), dtype='complex64')
    fv[0,:] = 1

    fv[1:(ssp_dim + 1) // 2,:] = phi
    fv[-1:ssp_dim // 2:-1,:] = -fv[1:(ssp_dim + 1) // 2,:]
    
    if ssp_dim % 2 == 0:
        fv[ssp_dim // 2,:] = 1

    return fv

class SSPEncoder:
    def __init__(self, phase_matrix, length_scale):
        '''
        Represents a domain using spatial semantic pointers.

        Parameters:
        -----------

        phase_matrix : np.ndarray
            A ssp_dim x domain_dim ndarray representing the frequency 
            components of the SSP representation.

        length_scale : float or np.ndarray
            Scales values before encoding.
        '''
        self.phase_matrix = phase_matrix

        self.domain_dim = self.phase_matrix.shape[1]
        self.ssp_dim = self.phase_matrix.shape[0]
        self.update_lengthscale(length_scale)

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
        
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ scaled_x.T), axis=0 ).real
        
        return data.T

def RandomSSPSpace(domain_dim, ssp_dim, length_scale = None, 
                   rng = np.random.default_rng(), psd_sampling = 'uniform' ):
    
    phase_matrix = make_unitary_matrix_fourier(ssp_dim,domain_dim, psd_sampling = psd_sampling )

    if isinstance(length_scale,float):
        length_scale = np.array( np.tile(length_scale,domain_dim) )
    else:
        length_scale = np.array( length_scale )
    return SSPEncoder(phase_matrix, length_scale=length_scale)

def make_model(param):

    with nengo.Network() as model:   
    
        def PID(e):
            return param.kp * e[0] + param.kd * e[1]
            
        #Nodes
        sys = nengo.Node(pen(dt = param.dt,m = param.m,g = param.g,b = param.b, l = param.l, t_dist = param.t_dist),size_in = 3)
        stim = nengo.Node( lambda t: [0.5*np.sin(t),0.5*np.cos(t)] )
        #stim = nengo.Node( lambda t: [1.,0.] if t < 0.1 else 0. )
        
        #Ensembles
        err = nengo.Ensemble(param.num_neurons,2)
        con = nengo.Ensemble(param.num_neurons,1)
        integral = nengo.Ensemble(param.num_neurons,1)

        #Connections
        nengo.Connection(err[0],integral,transform = 0.1) #Integral 
        nengo.Connection(integral,integral)
        nengo.Connection(integral, con,transform = param.ki)
        
        nengo.Connection(sys,err,transform = -1,synapse  = param.tau_s) #Error
        nengo.Connection(stim,err,synapse  = param.tau_s)
        
        nengo.Connection(sys,sys[0:2],synapse  = param.tau_s) #State Memory
        
        nengo.Connection(err,con,function = PID,synapse  = param.tau_s) #Control
        nengo.Connection(con, sys[2],transform = 10,synapse  = param.tau_s)
        
        #Adaptive Controller Block
        if param.adaptive_controller == True:
            
            if param.use_ssp == False:
                model.adaptive = nengo.Ensemble(param.num_neurons,
                                            dimensions = 2,
                                            radius = param.radius,
                                            #bias = param.bias + np.zeros(param.num_neurons),
                                            #gain = param.gain * np.ones(param.num_neurons)
                                            )
                nengo.Connection(sys,model.adaptive)
                ssp_embedding = None
            else:
                if param.ssp_type == 'random':
                    print('generating ssp space')
                    ssp_embedding = RandomSSPSpace( domain_dim = 2, 
                                                       ssp_dim = param.ssp_dim, 
                                                       length_scale = np.array([[param.ssp_ls1,param.ssp_ls2]]), 
                                                       psd_sampling = param.psd_sampling )
                
                elif param.ssp_type == 'hexssp':
                    ssp_embedding = sspspace.HexagonalSSPSpace(
                                            domain_dim = 2, 
                                            length_scale = np.atleast_2d([param.ssp_ls1,param.ssp_ls2]).T,
                                            n_rotates = param.n_rotates, 
                                            n_scales = param.n_scales,
                                            scale_min = param.scale_min, 
                                            scale_max = param.scale_max,
                                            )
                
                def encode_ssp(t,x):
                    return ssp_embedding.encode(x).flatten()

                if param.encoders_type == 'random':
                    encoders = nengo.dists.UniformHypersphere(surface=True).sample(param.num_neurons, param.ssp_dim)
                
                elif param.encoders_type == 'place-cells':
                    e_xs = np.random.uniform(low=param.domain_ranges_[:,0],high=param.domain_ranges_[:,1],size=(param.num_neurons,ssp_embedding.domain_dim))
                    encoders = ssp_embedding.encode(e_xs)
                    
                elif param.encoders_type == 'grid-cells':
                    encoders = ssp_embedding.sample_grid_encoders(param.num_neurons, param.seed)
                    
                xi = - sparsity_to_x_intercept( d = ssp_embedding.ssp_dim, p = param.rho_specified )
                print('xi: ', xi)
                
                # create ensemble; random encoders for now
                model.adaptive = nengo.Ensemble(n_neurons = param.num_neurons, 
                                          dimensions = ssp_embedding.ssp_dim,
                                          gain = param.gain * np.ones(param.num_neurons),
                                          bias = np.zeros(param.num_neurons) + xi,
                                          neuron_type = param.neuron_type,
                                          encoders = encoders,
                                          normalize_encoders = True,
                                         )
                ssp_node = nengo.Node(encode_ssp,size_in=2)
                nengo.Connection(sys[:2],ssp_node)
                nengo.Connection(ssp_node,model.adaptive)
            
            a = nengo.Connection(model.adaptive,con,transform = np.zeros((1,model.adaptive.dimensions)),learning_rule_type=nengo.PES(learning_rate=param.learning_rate))
            nengo.Connection(err,a.learning_rule,function = PID,transform = -1)  
        
        #Probes
        model.sys_probe = nengo.Probe(target = sys, attr = "output")
        model.stim_probe = nengo.Probe(target = stim, attr = "output")
    
    sim = nengo.Simulator(model,dt=param.dt,progress_bar=False)

    return model,sim,ssp_embedding

class AdaptiveControllerTrial(pytry.Trial):
    def params(self):
        
        # System parameters
        self.param( 'Mass of pendulum', m = 1. )
        self.param( 'Gravity of pendulum environment', g = 9.8 )
        self.param( 'Length of pendulum', l = 1. )
        self.param( 'Scale on input', b = 0.05 )
        self.param( 'Time of disturbance onset', t_dist = 5. )
        self.param( 'Domain bounds', domain_ranges_ = np.array([[-np.pi,np.pi],[-10.,10.]]) )
#        self.param( 'Domain bounds', domain_ranges_ = np.array([[-1.,1.],[-1.,1.]]) )
         
        # Simulation parameters
        self.param( 'Simulation timestep', dt = 0.001 )
        self.param( 'Simulation run time', sim_runtime = 20. )
        
        # Controller parameters
        self.param( 'Control gain for proportional error term', kp = 10. )
        self.param( 'Control gain for derivative error term', kd = 10. )
        self.param( 'Control gain for integral error term', ki = 0. )
        
        self.param( 'Toggle adaptive controller', adaptive_controller = True )
        self.param( 'Learning rate of adaptive controller', learning_rate = 1e-4 )
        
        # SSP representation parameters
        self.param( 'Whether or not to use SSP', use_ssp = True )
        self.param( 'Type of SSP', ssp_type = 'random' )
        self.param( 'Dimensionality of the random SSP representation', ssp_dim = 128 )
        self.param( 'Power spectral density sampling for random SSP', psd_sampling = 'uniform' )
        self.param( 'Length scale of SSP representation for variable 1', ssp_ls1 = 0.1 )
        self.param( 'Length scale of SSP representation for variable 2', ssp_ls2 = 0.1 )
        self.param( 'Sparsity of hidden layer', rho_specified = 0.1 )
        self.param( 'Sampling of SSP space by neurons', encoders_type = 'place-cells' )
        self.param( 'Number of rotates for the HexSSP representation', n_rotates = 5 )
        self.param( 'Number of scales for the HexSSP representation', n_scales = 8 )
        self.param( 'Minimum scale for the HexSSP representation', scale_min = 0.1 )
        self.param( 'Maximum scale for the HexSSP representation', scale_max = 3. )
        
        # Hidden layer parameters
        self.param( 'Number of neurons', num_neurons = 100 )
        self.param( 'Synapse on neurons', tau_s = 0. )
        self.param( 'Gain on neuron', gain = 1. )
        self.param( 'Bias on ReLU neuron', bias = 1. )
        self.param( 'Radius of the population', radius = np.sqrt(2) )
        self.param( 'Neuron type', neuron_type = nengo.SpikingRectifiedLinear() )
        
        # Experiment parameters
        self.param( 'Plot results', plot = True )
        self.param( 'NNI id', nni_id = None )
        self.param( 'NNI expt', nni_expt = None )
        
    def evaluate(self,param):
        np.random.seed(param.seed)
        model,sim,ssp_embedding = make_model(param)
        sim.run(param.sim_runtime)
    
        ref_signal = sim.data[model.stim_probe]
        out_signal = sim.data[model.sys_probe]
        
        t_idx_dist = int(param.t_dist / param.dt)
        mse = mean_squared_error(ref_signal[t_idx_dist:,0],out_signal[t_idx_dist:,0])        
        
        n_eval_points = 100
        meshes = np.meshgrid(*[np.linspace(b[0], b[1], n_eval_points) 
                                for b in param.domain_ranges_])
        eval_xs = np.vstack([m.flatten() for m in meshes]).T

        if param.use_ssp == True:
            eval_phis = np.array(ssp_embedding.encode(eval_xs))
            _,A = nengo.utils.ensemble.tuning_curves( ens = model.adaptive, sim = sim, inputs = eval_phis )
        else:
            _,A = nengo.utils.ensemble.tuning_curves( ens = model.adaptive, sim = sim, inputs = eval_xs )

        rho_actual = ( A > 0 ).mean()
        
        if param.plot == True:

            #Plot and evaluate
            t = sim.trange()

            if param.use_ssp == True:
                title = 'With SSP embedding'
            else:
                title = 'No SSP embedding'

            fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(4.,12.),sharex=True)
            ax1.plot(t, ref_signal[:,0], label='Ref Signal',zorder=10,color='k',linestyle='--')
            ax1.plot(t, out_signal[:,0], label='State Signal')
            ax1.legend(loc='lower left')
            ax1.set_ylabel('Amplitude')

            ax2.plot(t, ref_signal[:,1], label='Ref Signal',zorder=10,color='k',linestyle='--')
            ax2.plot(t, out_signal[:,1], label='State Signal')
            ax2.legend(loc='lower left')

            ax3.plot(t, ref_signal[:,0]-out_signal[:,0],color='dimgray',alpha=0.1)
            ax3.axhline(0.,color='k')
            #ax3.set_yscale('log')
            ax3.set_ylim(-0.1,0.1)
            ax3.set_ylabel('Error')
            ax3.set_xlabel('Time (s)')
            
            for ax in (ax1,ax2,ax3):
                ax.axvline(param.t_dist,color='k',linestyle='--')
            
            fig.suptitle(title)
            plt.legend()
            plt.show()

            # plot normalized activities
            
            A /= A.max()
            fig,axs = plt.subplots(1,5,figsize=(15.,3.))
            neurons_to_show = np.random.choice(range(param.num_neurons),5,replace=False)
            for n,ax in zip(neurons_to_show,axs.ravel()):
                im = ax.imshow(A[:,n].reshape(n_eval_points,n_eval_points),
                              origin = 'lower',
                              extent = param.domain_ranges_.ravel(),
                              aspect = 'auto',
                              vmin = 0., vmax = 1.
                         )
                if ax == axs[0]:
                    ax.set_xlabel(r'$x$')
                    ax.set_ylabel(r'$\dot x$')
                else:
                    ax.axis('off')
            fig.colorbar(im,ax=ax,label='Firing Rate (a.u.)')
            fig.tight_layout()
            if param.use_ssp == True:
                filename = f'sspspace-{param.ssp_type}-{param.encoders_type}-examples.pdf'
            else:
                filename = f'statespace-examples.pdf'
            plt.savefig(filename,format='pdf')
            plt.show()
        
        return {
            'mean-squared-error' : mse,
            'sparsity-actual'    : rho_actual,
        }
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nni', type = bool, default = False ) 

if __name__ == '__main__':
    args = parser.parse_args()
    if args.nni == True:
        params = {}
        nni_expt = nni.get_experiment_id()
        nni_id = nni.get_trial_id()
        params.update( { 'nni_id'     : nni_id } )
        params.update( { 'nni_expt'   : nni_expt } ) 

        dtstr = datetime.now().strftime('%d_%m_%Y') 
        data_dir = f'data/hpo-ssp-random-{dtstr}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        
        params.update( {'data_dir'   : data_dir  } )
        params.update( {'data_format': 'npz'     } )
        params.update( {'plot'       : False     } )
        params.update( {'verbose'    : True      } )

        nni_params = nni.get_next_parameter()
        params.update( nni_params )        

        adc = AdaptiveControllerTrial()
        
        mse = 0.
        seeds = [1,7,42]
        for seed in seeds:
            metadata = adc.run(
                    seed = int(seed),
                    ** params
                )
            mse += metadata['mean-squared-error']
            
        nni.report_final_result( mse / len(seeds) )
    else:
        params = {}
        dtstr = datetime.now().strftime('%d_%m_%Y') 
        data_dir = f'data/debug-{dtstr}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        
        params.update( {'data_dir'   : data_dir  } )
        params.update( {'data_format': 'npz'     } )
        params.update( {'plot'       : True     } )
        params.update( {'verbose'    : True      } )
        
        adc = AdaptiveControllerTrial()
        metadata = adc.run(
                    seed = 0,
                    ** params
                    )