from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn
from .conversions import detector2source_jacobian, detector2source, detector2source_jacobian_q, detector2source_jacobian_single_mass
from scipy.stats import gaussian_kde
from .wrappers import modgravity_wrappers, lcdm_wrappers

class CBC_rate_mchirp_q(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame
    masses, spin parameters and mass ratio, rate evolution times differential of comoving volume. Source-frame mass distribution,
    spin distribution and redshift distribution are summed to be independent from each other.
    The wrapper works with luminosity distances and detector frame masses and optionally with some chosen spin parameters, used to compute the rate.
    Parameters
    ----------
    cosmology_wrapper: class
        Wrapper for the cosmological model
    mass_wrapper: class
        Wrapper for the source-frame mass distribution
    q_wrapper: class
        Wrapper for the mass ratio distribution
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    def __init__(self,cosmology_wrapper,mass_wrapper,
                 q_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):

        self.cw = cosmology_wrapper
        self.mw = mass_wrapper
        self.qw = q_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free

        if scale_free:
            self.population_parameters = self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters+self.qw.population_parameters
        else:
            self.population_parameters = self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters+self.qw.population_parameters + ['R0']

        event_parameters = ['chirp_mass', 'mass_ratio', 'luminosity_distance']

        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()

    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.qw.update(**{key: kwargs[key] for key in self.qw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})

        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})

        if not self.scale_free:
            self.R0 = kwargs['R0']

    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)

        z = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        ms1 = kwargs['chirp_mass']/(1.+z)
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)

        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1)+self.qw.log_pdf(kwargs['mass_ratio'])+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian_q(z,self.cw.cosmology))-xp.log1p(z)
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})

        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights

        return log_out

    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)

        z = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        ms1 = kwargs['chirp_mass']/(1.+z)
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)

        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1)+self.qw.log_pdf(kwargs['mass_ratio'])+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian_q(z,self.cw.cosmology))-xp.log1p(z)

        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})

        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights

        return log_out


class CBC_mixte_pop_rate(object):
    '''
    A rate class for mixture population models. The overall population is constructed as 

    .. math::
        p(\\theta) = \\lambda p_A(\\theta) + (1-\\lambda) \\lambda p_B(\\theta) 

    Parameters
    ----------
    rate1: class
        Icarogw rate class for the first population model
    rate2: class
        Icarogw rate class for the second population model
    common_parameters: list
        A list of strings with the parameters in common to the two models. E.g. R0, H0, Om0
    '''
    def __init__(self,rate1, rate2, common_parameters):
        mapping_1 = {}
        mapping_2 = {}
        self.rate1 = rate1
        self.rate2 = rate2
        self.lambda_pop = ['lambda_pop']
        self.scale_free = self.rate1.scale_free

        # Fill up the dict mapping1 with all non common param, adding the _pop1 tag at the end
        for param in self.rate1.population_parameters:
            if param not in common_parameters:
                mapping_1[param+'_pop1'] = param

        # Fill up the dict mapping2 with all non common param, adding the _pop2 tag at the end
        for param in self.rate2.population_parameters:
            if param not in common_parameters:
                mapping_2[param+'_pop2'] = param

        # Create the list of strings with the population parameters from mapping1, mapping2 and common param and the mixing param
        self.population_parameters = list(mapping_1.keys())+list(mapping_2.keys())+self.lambda_pop+common_parameters

        # Add the common param to both mapping1 and mapping2
        for param in common_parameters:
            mapping_1[param]=param
            mapping_2[param]=param
        
        # Create the self of mapping1, mapping2 and event_parameters (m1s,m2s,z)
        self.mapping_1=mapping_1
        self.mapping_2=mapping_2
        self.event_parameters = self.rate1.PEs_parameters
        
        # Create the PEs and Injections param self
        self.PEs_parameters = self.event_parameters.copy()
        self.injections_parameters = self.rate1.injections_parameters.copy()

    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        # Update both rates using their parameters from mapping1 and mapping2
        self.rate1.update(**{self.mapping_1[key]:kwargs[key] for key in self.mapping_1})
        self.rate2.update(**{self.mapping_2[key]:kwargs[key] for key in self.mapping_2})

        # Update the mixing parameter
        self.lambda_pop = kwargs['lambda_pop']
    
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''

        log_rate_PE_1 = self.rate1.log_rate_PE(prior,**kwargs)
        log_rate_PE_2 = self.rate2.log_rate_PE(prior,**kwargs)

        xp = get_module_array(log_rate_PE_2)

        toret = xp.logaddexp(xp.log(self.lambda_pop)+log_rate_PE_1, xp.log(1-self.lambda_pop)+log_rate_PE_2)
        return toret

    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        log_rate_injections_1 = self.rate1.log_rate_injections(prior,**kwargs)
        log_rate_injections_2 = self.rate2.log_rate_injections(prior,**kwargs)

        xp = get_module_array(log_rate_injections_2)

        toret = xp.logaddexp(xp.log(self.lambda_pop)+log_rate_injections_1,xp.log(1-self.lambda_pop)+log_rate_injections_2)
        return toret




# LVK Reviewed
class CBC_catalog_vanilla_rate_skymap(object):
    
    def __init__(self,catalog,cosmology_wrapper,rate_wrapper,scale_free=False):
        '''
        A wrapper for the CBC rate model that make use of of just the luminosity distance and position of GW events.
        Useful if you generate PE from skymaps

        Parameters
        ----------
        catalog: object
            icarogw catalog class containing the preprocessed galaxy catalog
        cosmology_wrapper: object
            Cosmology wrapper from icarogw
        rate_wrapper: object
            Merger rate wrapper from icarogw
        '''
        
        self.catalog = catalog
        self.cw = cosmology_wrapper
        self.rw = rate_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters =  self.cw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters =  self.cw.population_parameters+self.rw.population_parameters + ['Rgal']
            
        event_parameters = ['luminosity_distance','sky_indices']
        
        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})

        if self.cw.__class__.__name__ in modgravity_wrappers:
            self.cw_bgwrap = self.cw.bgwrap
        elif self.cw.__class__.__name__ in lcdm_wrappers:
            self.cw_bgwrap = self.cw
        else:
            raise ValueError('Please pass a LCDM or Mod gravity wrapper')
        
        self.catalog.sch_fun.build_MF(self.cw_bgwrap.cosmology)

        
            
            
        if not self.scale_free:
            self.Rgal = kwargs['Rgal']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)
        sx = get_module_array_scipy(prior)
        
        z = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        dNgal_cat,dNgal_bg=self.catalog.effective_galaxy_number_interpolant(z,kwargs['sky_indices'],self.cw_bgwrap.cosmology,average=False)

        # Effective number density of galaxies (Eq. 2.19 on the overleaf document)
        dNgaleff=dNgal_cat+dNgal_bg
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.rw.rate.log_evaluate(z)+xp.log(dNgaleff) \
        -xp.log1p(z)-xp.log(xp.abs(self.cw.cosmology.ddl_by_dz_at_z(z)))-xp.log(prior)
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.Rgal)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        FIX-ME this method should be made consistent with the galaxy catalog below
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        xp = get_module_array(prior)
        sx = get_module_array_scipy(prior)
        
        z = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        
        # We assume the galaxy catalog empty to apply completeness correction
        dNgaleff=self.catalog.sch_fun.background_effective_galaxy_density(-xp.inf*xp.ones_like(z),z)*self.cw.cosmology.dVc_by_dzdOmega_at_z(z)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.rw.rate.log_evaluate(z)+xp.log(dNgaleff) \
        -xp.log1p(z)-xp.log(xp.abs(self.cw.cosmology.ddl_by_dz_at_z(z)))-xp.log(prior)
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.Rgal)
        else:
            log_out = log_weights
            
        return log_out

# LVK Reviewed
class CBC_low_latency_skymap_EM_counterpart(object):
    def __init__(self,cosmology_wrapper,rate_wrapper, list_of_skymaps,scale_free=False):

        '''
        A wrapper for the CBC rate model that make use of LVK low latency skymaps and EM counterparts.
        Posterior samples are going to be possible EM counterparts (1 for each event). These EM PEs
        are then combined with the GW skymap

        Parameters
        ----------
        cosmology_wrapper: object
            Cosmology wrapper from icarogw
        rate_wrapper: object
            Merger rate wrapper from icarogw
        list_of_skymaps: object
            A list of icarogw skymaps objects, order must be compatible with catalog of EM counterparts passed in posterior sampels
        scale_free: True
            Scale free model or not
        '''
        
        self.cw = cosmology_wrapper
        self.rw = rate_wrapper
        self.scale_free = scale_free
        self.list_of_skymaps=list_of_skymaps
         
        if scale_free:
            self.population_parameters =  self.cw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters =  self.cw.population_parameters+self.rw.population_parameters + ['R0']
            
        self.PEs_parameters = ['z_EM','right_ascension','declination']
        self.injections_parameters = ['luminosity_distance']

        
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        xp = get_module_array(prior)
        sx = get_module_array_scipy(prior)
        
        if len(kwargs['z_EM'].shape) != 2:
            raise ValueError('The EM counterpart rate wants N_ev x N_samples arrays')

        dl_samples = self.cw.cosmology.z2dl(kwargs['z_EM'])       
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(kwargs['z_EM'])*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.rw.rate.log_evaluate(kwargs['z_EM'])+log_dVc_dz-xp.log(prior)-xp.log1p(kwargs['z_EM'])       
        
        n_ev = kwargs['z_EM'].shape[0]
        lwtot = xp.empty(kwargs['z_EM'].shape)
        for i in range(n_ev): 
            self.list_of_skymaps[i].intersect_EM_PE(kwargs['right_ascension'][i,:],kwargs['declination'][i,:])
            log_l_skymap = xp.log(self.list_of_skymaps[i].evaluate_3D_likelihood_intersected(dl_samples[i,:]))            
            lwtot[i,:] = sx.special.logsumexp(log_weights[i,:]+log_l_skymap)-xp.log(kwargs['z_EM'].shape[1])

        if not self.scale_free:
            log_out = lwtot + xp.log(self.R0)
        else:
            log_out = lwtot
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        xp = get_module_array(prior)
       
        z = self.cw.cosmology.dl2z(kwargs['luminosity_distance']) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(xp.abs(self.cw.cosmology.ddl_by_dz_at_z(z)))-xp.log1p(z)
        
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out

# LVK Reviewed
class CBC_vanilla_EM_counterpart(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame
    masses, spin parameters and redshift rate evolution times differential of comoving volume. Source-frame mass distribution,
    spin distribution and redshift distribution are summed to be independent from each other.

    .. math::
        \\frac{d N_{\\rm CBC}(\\Lambda)}{d\\vec{m}d\\vec{\\chi} dz dt_s} = R_0 \\psi(z;\\Lambda) p_{\\rm pop}(\\vec{m},\\vec{\\chi}|\\Lambda) \\frac{d V_c}{dz}

    The wrapper works with luminosity distances and detector frame masses and optionally with some chosen spin parameters.

    Note that this rate model also takes into account for the GW events an additional weight given by the EM counterpart, we defer to section 2.3 for more details.
    Note also that for evaluating selection biases, we do not account for biases given by EM observatories.

    Parameters
    ----------
    cosmology_wrapper: class
        Wrapper for the cosmological model
    mass_wrapper: class
        Wrapper for the source-frame mass distribution
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    def __init__(self,cosmology_wrapper,mass_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):
        self.cw = cosmology_wrapper
        self.mw = mass_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters + ['R0']

        event_parameters = ['mass_1', 'mass_2', 'luminosity_distance','z_EM']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
        self.injections_parameters.remove('z_EM')
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        xp = get_module_array(prior)
        sx = get_module_array_scipy(prior)
        
        if len(kwargs['mass_1'].shape) != 2:
            raise ValueError('The EM counterpart rate wants N_ev x N_samples arrays')
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
        
        n_ev = kwargs['mass_1'].shape[0]
        lwtot = xp.empty(kwargs['z_EM'].shape)
        for i in range(n_ev): 
            ww = xp.exp(log_weights[i,:])
            kde_fit = gaussian_kde(z[i,:],weights=ww/ww.sum())   
            lwtot[i,:] = sx.special.logsumexp(log_weights[i,:])-xp.log(kwargs['mass_1'].shape[1])+np2cp(kde_fit.logpdf(cp2np(kwargs['z_EM'][i,:])))

        if not self.scale_free:
            log_out = lwtot + xp.log(self.R0)
        else:
            log_out = lwtot
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        xp = get_module_array(prior)
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out

class CBC_rate_m1_given_redshift_q(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame primary mass , spin parameters and mass ratio, rate evolution times differential of comoving volume. Primary mass can be redshift dependent.
    The wrapper works with luminosity distances and detector frame masses and optionally with some chosen spin parameters, used to compute the rate.

    Parameters
    ----------
    cosmology_wrapper: class
        Wrapper for the cosmological model
    mass_wrapper: class
        Wrapper for the source-frame mass distribution
    q_wrapper: class
        Wrapper for the mass ratio distribution
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    def __init__(self,cosmology_wrapper,mass_redshift_wrapper,
                 q_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):
        
        self.cw = cosmology_wrapper
        self.m1zw = mass_redshift_wrapper
        self.qw = q_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters = self.cw.population_parameters+self.m1zw.population_parameters+self.rw.population_parameters+self.qw.population_parameters
        else:
            self.population_parameters = self.cw.population_parameters+self.m1zw.population_parameters+self.rw.population_parameters+self.qw.population_parameters + ['R0']
            
        event_parameters = ['mass_1', 'mass_ratio', 'luminosity_distance']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(  **{key: kwargs[key] for key in self.cw.population_parameters})
        self.m1zw.update(**{key: kwargs[key] for key in self.m1zw.population_parameters})
        self.qw.update(  **{key: kwargs[key] for key in self.qw.population_parameters})
        self.rw.update(  **{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)

        z           = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        # Convert m1 from detector to source frame.
        ms1         = kwargs['mass_1'] / (1.+z)
        log_dVc_dz  = xp.log( self.cw.cosmology.dVc_by_dzdOmega_at_z(z) * 4*xp.pi )
        
        # Compute the weights for the samples Monte Carlo integral, Eq.3 of [2305.17973].
        # w = 1/prior_d dN/(dm1d dq ddL dtd) = 1/prior_d dN/(dm1s dq dVc dts) 1/|J_d->s| 1/1+z dVc/dz
        # dN/(dm1s dq dVc dts) = R(z) p_pop(m1s|z) p_pop(q|m1s,z)
        log_weights = self.m1zw.log_pdf(ms1, z) + self.qw.log_pdf(kwargs['mass_ratio']) + self.rw.rate.log_evaluate(z) + log_dVc_dz \
        - xp.log(prior) - xp.log(detector2source_jacobian_q(z, self.cw.cosmology)) - xp.log1p(z)
        
        if self.sw is not None:
            log_weights += self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)
        
        z           = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        # Convert m1 from detector to source frame.
        ms1         = kwargs['mass_1'] / (1.+z)
        log_dVc_dz  = xp.log( self.cw.cosmology.dVc_by_dzdOmega_at_z(z) * 4*xp.pi )
        
        # Compute the weights for the samples Monte Carlo integral, Eq.7 of [2305.17973].
        # w = 1/prior_d dN/(dm1d dq ddL dtd) = 1/prior_d dN/(dm1s dq dVc dts) 1/|J_d->s| 1/1+z dVc/dz
        # dN/(dm1s dq dVc dts) = R(z) p_pop(m1s|z) p_pop(q|m1s,z)
        log_weights = self.m1zw.log_pdf(ms1, z) + self.qw.log_pdf(kwargs['mass_ratio']) + self.rw.rate.log_evaluate(z) + log_dVc_dz \
        - xp.log(prior) - xp.log(detector2source_jacobian_q(z, self.cw.cosmology)) - xp.log1p(z)
        
        if self.sw is not None:
            log_weights += self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out

class CBC_rate_m_given_redshift(object):
    def __init__(self,cosmology_wrapper,mass_redshift_wrapper,
                rate_wrapper,spin_wrapper=None,scale_free=False):
        
        self.cw = cosmology_wrapper
        self.mzw = mass_redshift_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters = self.cw.population_parameters+self.mzw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters = self.cw.population_parameters+self.mzw.population_parameters+self.rw.population_parameters + ['R0']
            
        event_parameters = ['mass_1', 'luminosity_distance']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update( **{key: kwargs[key] for key in self.cw.population_parameters})
        self.mzw.update(**{key: kwargs[key] for key in self.mzw.population_parameters})
        self.rw.update( **{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)

        z           = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        # Convert m from detector to source frame.
        ms          = kwargs['mass_1'] / (1.+z)
        log_dVc_dz  = xp.log( self.cw.cosmology.dVc_by_dzdOmega_at_z(z) * 4*xp.pi )
        
        # Compute the weights for the samples Monte Carlo integral, Eq.3 of [2305.17973].
        # w = 1/prior_d dN/(dmd ddL dtd) = 1/prior_d dN/(dms dVc dts) 1/|J_d->s| 1/1+z dVc/dz
        # dN/(dms dVc dts) = R(z) p_pop(ms|z)
        log_weights = self.mzw.log_pdf(ms, z) + self.rw.rate.log_evaluate(z) + log_dVc_dz \
        - xp.log(prior) - xp.log(detector2source_jacobian_single_mass(z, self.cw.cosmology)) - xp.log1p(z)
        
        if self.sw is not None:
            log_weights += self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)
        
        z           = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        # Convert m from detector to source frame.
        ms          = kwargs['mass_1'] / (1.+z)
        log_dVc_dz  = xp.log( self.cw.cosmology.dVc_by_dzdOmega_at_z(z) * 4*xp.pi )
        
        # Compute the weights for the samples Monte Carlo integral, Eq.7 of [2305.17973].
        # w = 1/prior_d dN/(dmd ddL dtd) = 1/prior_d dN/(dms dVc dts) 1/|J_d->s| 1/1+z dVc/dz
        # dN/(dms dVc dts) = R(z) p_pop(ms|z)
        log_weights = self.mzw.log_pdf(ms, z) + self.rw.rate.log_evaluate(z) + log_dVc_dz \
        - xp.log(prior) - xp.log(detector2source_jacobian_single_mass(z, self.cw.cosmology)) - xp.log1p(z)
        
        if self.sw is not None:
            log_weights += self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out


class CBC_rate_total_mass_q(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame
    total mass, spin parameters and mass ratio, rate evolution times differential of comoving volume. Source-frame mass distribution,
    spin distribution and redshift distribution are summed to be independent from each other.

    The wrapper works with luminosity distances and detector frame masses and optionally with some chosen spin parameters, used to compute the rate.

    Parameters
    ----------
    cosmology_wrapper: class
        Wrapper for the cosmological model
     mass_redshift_wrapper: class
        Primary mass wrapper conditioned on redshift
    q_wrapper: class
        Wrapper for the mass ratio distribution
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    
    def __init__(self,cosmology_wrapper,mass_wrapper,
                 q_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):
        
        self.cw = cosmology_wrapper
        self.mw = mass_wrapper
        self.qw = q_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters = self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters+self.qw.population_parameters
        else:
            self.population_parameters = self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters+self.qw.population_parameters + ['R0']
            
        event_parameters = ['total_mass', 'mass_ratio', 'luminosity_distance']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.qw.update(**{key: kwargs[key] for key in self.qw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)

        z = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        ms1 = kwargs['total_mass']/(1.+z)
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1)+self.qw.log_pdf(kwargs['total_mass'])+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian_q(z,self.cw.cosmology))-xp.log1p(z)
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights

        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        return self.log_rate_PE(prior,**kwargs)


class CBC_rate_m1_q(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame
    masses, spin parameters and mass ratio, rate evolution times differential of comoving volume. Source-frame mass distribution,
    spin distribution and redshift distribution are summed to be independent from each other.

    The wrapper works with luminosity distances and detector frame masses and optionally with some chosen spin parameters, used to compute the rate.

    Parameters
    ----------
    cosmology_wrapper: class
        Wrapper for the cosmological model
    mass_wrapper: class
        Wrapper for the source-frame mass distribution
    q_wrapper: class
        Wrapper for the mass ratio distribution
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    def __init__(self,cosmology_wrapper,mass_wrapper,
                 q_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):
        
        self.cw = cosmology_wrapper
        self.mw = mass_wrapper
        self.qw = q_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters = self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters+self.qw.population_parameters
        else:
            self.population_parameters = self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters+self.qw.population_parameters + ['R0']
            
        event_parameters = ['mass_1', 'mass_ratio', 'luminosity_distance']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.qw.update(**{key: kwargs[key] for key in self.qw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)

        z           = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        # Convert m1 from detector to source frame.
        ms1         = kwargs['mass_1'] / (1.+z)
        log_dVc_dz  = xp.log( self.cw.cosmology.dVc_by_dzdOmega_at_z(z) * 4*xp.pi )
        
        # Compute the weights for the samples Monte Carlo integral, Eq.3 of [2305.17973].
        # w = 1/prior_d dN/(dm1d dq ddL dtd) = 1/prior_d dN/(dm1s dq dVc dts) 1/|J_d->s| 1/1+z dVc/dz
        # dN/(dm1s dq dVc dts) = R(z) p_pop(m1s) p_pop(q|m1s)
        log_weights = self.mw.log_pdf(ms1) + self.qw.log_pdf(kwargs['mass_ratio']) + self.rw.rate.log_evaluate(z) + log_dVc_dz \
        - xp.log(prior) - xp.log(detector2source_jacobian_q(z, self.cw.cosmology)) - xp.log1p(z)

        if self.sw is not None:
            log_weights += self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights

        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)
        
        z           = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        # Convert m1 from detector to source frame.
        ms1         = kwargs['mass_1'] / (1.+z)
        log_dVc_dz  = xp.log( self.cw.cosmology.dVc_by_dzdOmega_at_z(z) * 4*xp.pi )
        
        # Compute the weights for the samples Monte Carlo integral, Eq.3 of [2305.17973].
        # w = 1/prior_d dN/(dm1d dq ddL dtd) = 1/prior_d dN/(dm1s dq dVc dts) 1/|J_d->s| 1/1+z dVc/dz
        # dN/(dm1s dq dVc dts) = R(z) p_pop(m1s) p_pop(q|m1s)
        log_weights = self.mw.log_pdf(ms1) + self.qw.log_pdf(kwargs['mass_ratio']) + self.rw.rate.log_evaluate(z) + log_dVc_dz \
        - xp.log(prior) - xp.log(detector2source_jacobian_q(z, self.cw.cosmology)) - xp.log1p(z)
        
        if self.sw is not None:
            log_weights += self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    

class CBC_rate_m1_given_redshift_m2(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame
    masses, spin parameters and mass ratio, rate evolution times differential of comoving volume. The primary mass is assumed to be conditional dependent from the redshift

    The wrapper works with luminosity distances and detector frame masses and optionally with some chosen spin parameters, used to compute the rate.

    Parameters
    ----------
    cosmology_wrapper: class
        Wrapper for the cosmological model
    mass_redshift_wrapper: class
        Primary mass wrapper conditioned on redshift
    mass2_wrapper: class
        Mass wrapper for the secondary mass
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    def __init__(self,cosmology_wrapper,mass_redshift_wrapper,
                 mass2_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):
        
        self.cw = cosmology_wrapper
        self.m1zw = mass_redshift_wrapper
        self.m2w = mass2_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters = self.cw.population_parameters+self.m1zw.population_parameters+self.rw.population_parameters+self.m2w.population_parameters
        else:
            self.population_parameters = self.cw.population_parameters+self.m1zw.population_parameters+self.rw.population_parameters+self.m2w.population_parameters + ['R0']
            
        event_parameters = ['mass_1', 'mass_2', 'luminosity_distance']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.m1zw.update(**{key: kwargs[key] for key in self.m1zw.population_parameters})
        self.m2w.update(**{key: kwargs[key] for key in self.m2w.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)

        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        # FIXME: Need to marginalize for m2 distribution
        log_weights=self.m1zw.log_pdf(ms1,z)+self.m2w.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        # FIXME: Need to marginalize for m2 distribution
        log_weights=self.m1zw.log_pdf(ms1,z)+self.m2w.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out

# LVK Reviewed
class CBC_vanilla_rate(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame
    masses, spin parameters and redshift rate evolution times differential of comoving volume. Source-frame mass distribution,
    spin distribution and redshift distribution are summed to be independent from each other.

    .. math::
        \\frac{d N_{\\rm CBC}(\\Lambda)}{d\\vec{m}d\\vec{\\chi} dz dt_s} = R_0 \\psi(z;\\Lambda) p_{\\rm pop}(\\vec{m},\\vec{\\chi}|\\Lambda) \\frac{d V_c}{dz}

    The wrapper works with luminosity distances and detector frame masses and optionally with some chosen spin parameters, used to compute the rate.

    Parameters
    ----------
    cosmology_wrapper: class
        Wrapper for the cosmological model
    mass_wrapper: class
        Wrapper for the source-frame mass distribution
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    def __init__(self,cosmology_wrapper,mass_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):
        
        self.cw = cosmology_wrapper
        self.mw = mass_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters + ['R0']
            
        event_parameters = ['mass_1', 'mass_2', 'luminosity_distance']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out


class CBC_vanilla_rate_pseob(CBC_vanilla_rate):
    def __init__(self,cosmology_wrapper,mass_wrapper,rate_wrapper,pseob_wrapper,spin_wrapper=None,scale_free=False):

        super().__init__(cosmology_wrapper=cosmology_wrapper,
                         mass_wrapper=mass_wrapper,rate_wrapper=rate_wrapper,
                         spin_wrapper=spin_wrapper,scale_free=scale_free)

        self.pseobw = pseob_wrapper
        self.population_parameters = self.population_parameters + self.pseobw.population_parameters
                    
        self.PEs_parameters = self.PEs_parameters + ['domega220','dtau220']
        self.injections_parameters = self.injections_parameters + ['domega220','dtau220']
            
    def update(self,**kwargs):
        super().update(**kwargs)
        self.pseobw.update(**kwargs)
        
    def log_rate_PE(self,prior,**kwargs):
        return super().log_rate_PE(prior,**kwargs) + self.pseobw.log_pdf(kwargs['domega220'],kwargs['dtau220'])
    
    def log_rate_injections(self,prior,**kwargs):           
        return super().log_rate_injections(prior,**kwargs) + self.pseobw.log_pdf(kwargs['domega220'],kwargs['dtau220'])

class CBC_vanilla_rate_pseob_dummy(CBC_vanilla_rate):
    def __init__(self,cosmology_wrapper,mass_wrapper,rate_wrapper,pseob_wrapper,spin_wrapper=None,scale_free=False):

        super().__init__(cosmology_wrapper=cosmology_wrapper,
                         mass_wrapper=mass_wrapper,rate_wrapper=rate_wrapper,
                         spin_wrapper=spin_wrapper,scale_free=scale_free)

        self.pseobw = pseob_wrapper
        self.population_parameters = self.population_parameters + self.pseobw.population_parameters
                    
        self.PEs_parameters = self.PEs_parameters + ['domega220','dtau220']
        self.injections_parameters = self.injections_parameters
            
    def update(self,**kwargs):
        super().update(**kwargs)
        self.pseobw.update(**kwargs)
        
    def log_rate_PE(self,prior,**kwargs):
        return super().log_rate_PE(prior,**kwargs) + self.pseobw.log_pdf(kwargs['domega220'],kwargs['dtau220'])
    
    def log_rate_injections(self,prior,**kwargs):           
        return super().log_rate_injections(prior,**kwargs) 


class CBC_vanilla_rate_spins(CBC_vanilla_rate):
    def __init__(self,cosmology_wrapper,mass_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):
        
        super().__init__(cosmology_wrapper,mass_wrapper,
                         rate_wrapper,spin_wrapper=spin_wrapper,
                         scale_free=scale_free)
                
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters}
                                         ,mass_1_source=ms1,mass_2_source=ms2)
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        return self.log_rate_PE(prior,**kwargs)


# LVK Reviewed
class CBC_catalog_vanilla_rate(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame
    masses, spin parameters and redshift rate evolution times galaxy number density. This rate model also uses galaxy catalogs.
    Source-frame mass distribution, spin distribution and redshift distribution are summed to be independent from each other.

    .. math::
        \\frac{dN_{\\rm CBC}(\\Lambda)}{dz d\\vec{m} d\\vec{\\chi} d\\Omega dt_s} = R^{*}_{\\rm gal,0} \\psi(z;\\Lambda) p_{\\rm pop}(\\vec{m},  \\vec{\\chi}|\\Lambda) \\times 
        
    .. math::
        \\times \\left[ \\frac{dV_c}{dz d\\Omega} \\phi_*(H_0)\\Gamma_{\\rm inc}(\\alpha+\\epsilon+1,x_{\\rm max}(M_{\\rm thr}),x_{\\rm min}) + \\sum_{i=1}^{N_{\\rm gal}(\\Omega)} f_{L}(M(m_i,z);\\Lambda) p(z|z^i_{\\rm obs},\\sigma^i_{\\rm z,obs}) \\right],

    The wrapper works with luminosity distances, detector frame masses and sky pixels and optionally with some chosen spin parameters.

    Parameters
    ----------
    catalog: class
        Catalog class already processed to caclulate selection biases from the galaxy catalog.
    cosmology_wrapper: class
        Wrapper for the cosmological model
    mass_wrapper: class
        Wrapper for the source-frame mass distribution
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    def __init__(self,catalog,cosmology_wrapper,mass_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):
        
        self.catalog = catalog
        self.cw = cosmology_wrapper
        self.mw = mass_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters + ['Rgal']
            
        event_parameters = ['mass_1', 'mass_2', 'luminosity_distance','sky_indices']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})

        if self.cw.__class__.__name__ in modgravity_wrappers:
            self.cw_bgwrap = self.cw.bgwrap
        elif self.cw.__class__.__name__ in lcdm_wrappers:
            self.cw_bgwrap = self.cw
        else:
            raise ValueError('Please pass a LCDM or Mod gravity wrapper')
        
        self.catalog.sch_fun.build_MF(self.cw_bgwrap.cosmology)
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            
            self.Rgal = kwargs['Rgal']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology)
        dNgal_cat,dNgal_bg=self.catalog.effective_galaxy_number_interpolant(z,kwargs['sky_indices'],self.cw_bgwrap.cosmology,average=False)

        # Effective number density of galaxies (Eq. 2.19 on the overleaf document)
        dNgaleff=dNgal_cat+dNgal_bg
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+xp.log(dNgaleff) \
        -xp.log1p(z)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log(prior)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.Rgal)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        xp = get_module_array(prior)
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology)
        dNgal_cat,dNgal_bg=self.catalog.effective_galaxy_number_interpolant(z,kwargs['sky_indices'],self.cw_bgwrap.cosmology,average=True)

        # Effective number density of galaxies (Eq. 2.19 on the overleaf document)
        dNgaleff=dNgal_cat+dNgal_bg
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+xp.log(dNgaleff) \
        -xp.log1p(z)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log(prior)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.Rgal)
        else:
            log_out = log_weights
            
        return log_out
