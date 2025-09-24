import os as _os
import os as os
import numpy as np
from .cupy_pal import get_module_array

def write_condor_files(home_folder,uname='simone.mastrogiovanni',
agroup='ligo.dev.o4.cbc.hubble.icarogw',memory=10000,cpus=1,disk=10000):
    '''
    This function looks for all the *.py files in a folder and write a set of condor files
    needed for submission on write_condor_files. To launch the jobs, 1) Generate files with this function
    2) run chmod +x *.sh 3) launch the sub files.

    Parameters
    ----------
    home_folder: str
        Folder where to look for python files
    uname: str
        Username for condor
    agroup: str
        Accounting group for condor
    '''
    list_py_files = _os.listdir(home_folder)

    for file in list_py_files:
        if file.endswith('.py'):
            if file=='config.py':
                continue
            fname = file[:-3:1]

            f = open(home_folder+fname+'.sh', 'w')
            f.write('#!/bin/bash')
            f.write('\n')
            f.write('MYJOB_DIR='+home_folder)
            f.write('\n')
            f.write('cd ${MYJOB_DIR}')
            f.write('\n')
            f.write('python '+file)
            f.close()

            f = open(home_folder+fname+'.sub', 'w')
            f.write('universe = vanilla\n')
            f.write('getenv = True\n')
            f.write('executable = '+home_folder+fname+'.sh\n')
            f.write('accounting_group = '+agroup+'\n')
            f.write('accounting_group_user = '+uname)
            f.write('\n')
            f.write('request_memory ='+str(memory)+'\n')
            f.write('request_cpus ='+str(cpus)+'\n')
            f.write('request_disk ='+str(disk)+'\n')    
            f.write('output = '+home_folder+fname+'.stdout\n')
            f.write('error = '+home_folder+fname+'.stderr\n')
            f.write('log = '+home_folder+fname+'.log\n')
            f.write('Requirements = TARGET.Dual =!= True\n')
            f.write('queue\n')
            f.close()
            _os.system('chmod a+x '+home_folder+'*.sh')


def check_posterior_samples_and_prior(posterior_samples, prior):
    """
    This function asserts whether all entries of the posterior_samples
    dictionary have the same length as the prior array. It will also check if the prior has non-zero values.

    Parameters
    ----------
    posterior_samples: dict
        Dictionary of all parameters
    prior: array
        Array of the probability of the prior used for parameter estimation. 

    Returns: 
    --------
    
    None, if the test passes, otherwise, it raises an 
    error. 
    
    """

    # compute the number of prior samples
    n_prior = len(prior)
    for param in posterior_samples.keys():
        n_posterior_samples = len(posterior_samples[param])

        # throw an error if the length do not agree
        if(n_posterior_samples!=n_prior):
            print(f'{param} does not contain as many samples as the prior. ')
            raise ValueError
            
    xp = get_module_array(prior)
    idx_zero = prior==0.
    if xp.any(idx_zero):
        print('The zero values are in position ', xp.where(idx_zero)[0])
        raise ValueError('Prior can not have 0 values')
    return None

def write_condor_files_catalog(home_folder, outfolder, nside, uname='simone.mastrogiovanni', agroup='ligo.dev.o4.cbc.hubble.icarogw'):
    '''
    Writes the python scripts and condor files to pixelize a galaxy catalog.

    Parameters
    ----------
    home_folder: str
        Where to place the running scripts
    outfolder: str
        Where to place the pixelated catalog
    nside: int
        Nside of the healpy pixelization
    uname: str
        Username for condor
    agroup: str
        Accounting group for condor
    '''
    
    try:
        os.mkdir('logs')
    except:
        pass
        
    
    fp = open(os.path.join(home_folder,'make_pixel_files.py'),'w')
    fp.write('import icarogw \n')
    fp.write('import h5py \n')
    fp.write('outfolder = \'{:s}\' \n'.format(outfolder))
    fp.write('nside = {:d} \n'.format(nside))
    fp.write('with h5py.File(\'\',\'r\') as cat:\n')
    fp.write('\ticarogw.catalog.create_pixelated_catalogs(outfolder,nside,{key:cat[key] for key in [\'\']})\n')
    fp.write('icarogw.catalog.clear_empty_pixelated_files(outfolder,nside) \n')
    fp.close()
        
    f = open(os.path.join(home_folder,'make_pixel_files.sh'),'w')
    f.write('#!/bin/bash')
    f.write('\n')
    f.write('MYJOB_DIR='+home_folder)
    f.write('\n')
    f.write('cd ${MYJOB_DIR}')
    f.write('\n')
    f.write('python make_pixel_files.py')
    f.close()

    f = open(os.path.join(home_folder,'make_pixel_files.sub'),'w')
    f.write('universe = vanilla\n')
    f.write('getenv = True\n')
    f.write(f'executable = {home_folder}/make_pixel_files.sh \n')
    f.write('accounting_group = '+agroup+'\n')
    f.write('accounting_group_user = '+uname)
    f.write('\n')
    f.write('request_memory = 8G \n')
    f.write('request_cpus = 1 \n')
    f.write('request_disk = 10G \n')    
    f.write(f'output = {home_folder}/logs/make_pixel_files.stdout \n')
    f.write(f'error = {home_folder}/logs/make_pixel_files.stderr \n')
    f.write('Requirements = TARGET.Dual =!= True \n')
    f.write('queue\n')
    f.close()


def write_condor_files_nan_removal_mthr_computation(home_folder, outfolder, fields_to_take, grouping,apparent_magnitude_flag,nside_mthr,mthr_percentile,Nintegration,Numsigma,zcut,NumJobs,uname='simone.mastrogiovanni',
agroup='ligo.dev.o4.cbc.hubble.icarogw'):

    '''
    Writes the python scripts and condor files to removes NaNs and calculate apparent magnitude threshold cut for galaxy catalog

    Parameters
    ----------
    home_folder: str
        Where to place the running scripts
    outfolder: str
        Where to place the pixelated catalog
    fields_to_take: list
        list of strings containing the groups to take from the pixelated files
    grouping: str
        Name of the new group to create in the pixelated files to save the apparent mthr cut
    apparent_magnitude_flag: str
        String that indetifies the group of m to use for the mthr calculation
    nside_mthr: int
        Nside to use for calculating the apparent magnitude thr cut
    mthr_percentile: float
        Percentile of galaxies to use to define mthr, between 0-100
    Nintegration: int or np.array
        If int, it is the number of points taken to integrate the galaxy redshift uncertainties. If np.array, it is the redshift grid resolution
    Numsigma: int
        Number of sigmas to consider for the EM likelihood of galaxies
    zcut: float
        Redshift at which to cut the galaxy catalog
    NumJobs: int
        Number of jubs to run on condor
    uname: str
        Username for condor
    agroup: str
        Accounting group for condor
    '''

    filled_pixels = np.genfromtxt(os.path.join(outfolder,'filled_pixels.txt')).astype(int)
    NumPix = int(np.ceil(len(filled_pixels)/NumJobs))

    bot = np.arange(0,len(filled_pixels),NumPix)
    top = np.arange(NumPix,len(filled_pixels)+NumPix,NumPix)
    np.savetxt(os.path.join(home_folder,'queue_NaN.txt'),np.column_stack([bot,top]),fmt='%d')

    fp = open(os.path.join(home_folder,'clear_NaNs.py'),'w')
    fp.write('import icarogw \n')
    fp.write('import numpy as np \n')
    fp.write('import sys \n')
    fp.write('from tqdm import tqdm \n')
    fp.write('outfolder = \'{:s}\' \n'.format(outfolder))
    fp.write('fields_to_take = [')
    for i,ff in enumerate(fields_to_take):
        fp.write('\'{:s}\''.format(ff))
        if i!=(len(fields_to_take)-1):
            fp.write(', ')
    fp.write(']\n')
    fp.write('grouping = \'{:s}\' \n'.format(grouping))
    fp.write('bot_pix = int(sys.argv[1])\n')
    fp.write('top_pix = int(sys.argv[2])\n')
    fp.write('filled_pixels = np.genfromtxt(\'{:s}\').astype(int) \n'.format(
        os.path.join(outfolder,'filled_pixels.txt')))
    fp.write('filled_pixels = filled_pixels[bot_pix:top_pix] \n')
    fp.write('for pix in tqdm(filled_pixels,desc=\'Cleaning pixel\'):\n')
    fp.write('\ticarogw.catalog.remove_nans_pixelated_files(outfolder,pix,fields_to_take,grouping)\n')
    fp.close()

    f = open(os.path.join(home_folder,'clear_NaNs.sh'),'w')
    f.write('#!/bin/bash')
    f.write('\n')
    f.write('MYJOB_DIR='+home_folder)
    f.write('\n')
    f.write('cd ${MYJOB_DIR}')
    f.write('\n')
    f.write('python clear_NaNs.py $1 $2')
    f.close()

    f = open(os.path.join(home_folder,'clear_NaNs.sub'),'w')
    f.write('universe = vanilla\n')
    f.write('getenv = True\n')
    f.write(f'executable = {home_folder}/clear_NaNs.sh \n')
    f.write('accounting_group = '+agroup+'\n')
    f.write('accounting_group_user = '+uname)
    f.write('\n')
    f.write('request_memory = 8G \n')
    f.write('request_cpus = 1 \n')
    f.write('request_disk = 10G \n')    
    f.write('arguments = $(Item) $(Item2) \n')

    f.write(f'output = {home_folder}/logs/clear_nans_$(Item)_$(Item2).stdout \n')
    f.write(f'error = {home_folder}/logs/clear_nans_$(Item)_$(Item2).stderr \n')
    f.write('Requirements = TARGET.Dual =!= True \n')
    f.write('queue Item, Item2 from {:s}\n'.format(os.path.join(home_folder,'queue_NaN.txt')))
    f.close()


    fp = open(os.path.join(home_folder,'calc_mthr_and_grid.py'),'w')
    fp.write('import icarogw \n')
    fp.write('import numpy as np \n')
    fp.write('import sys \n')
    fp.write('from astropy.cosmology import Planck15 \n')
    fp.write('from tqdm import tqdm \n')
    fp.write('cosmo_ref = icarogw.cosmology.astropycosmology(zmax=10.)\n')
    fp.write('cosmo_ref.build_cosmology(Planck15)\n')
    fp.write('outfolder = \'{:s}\' \n'.format(outfolder))
    fp.write('grouping = \'{:s}\' \n'.format(grouping))
    fp.write('apparent_magnitude_flag = \'{:s}\' \n'.format(apparent_magnitude_flag))
    fp.write('nside_mthr =  {:d} \n'.format(nside_mthr))
    fp.write('mthr_percentile =  {:f} \n'.format(mthr_percentile))
    if isinstance(Nintegration,np.ndarray):
        fp.write('Nintegration =  np.logspace(np.log10({:f}),np.log10({:f}),{:d}) \n'.format(Nintegration.min(),Nintegration.max(),len(Nintegration)))
    else:
        fp.write('Nintegration =  {:d} \n'.format(Nintegration))
    fp.write('Numsigma =  {:d} \n'.format(Numsigma))
    fp.write('zcut =  {:f} \n'.format(zcut))
    fp.write('bot_pix = int(sys.argv[1])\n')
    fp.write('top_pix = int(sys.argv[2])\n')
    fp.write('filled_pixels = np.genfromtxt(\'{:s}\').astype(int) \n'.format(
        os.path.join(outfolder,'filled_pixels.txt')))
    fp.write('filled_pixels = filled_pixels[bot_pix:top_pix] \n')
    fp.write('for pix in tqdm(filled_pixels,desc=\'Calculating apparent magnitude\'):\n')
    fp.write('\ticarogw.catalog.calculate_mthr_pixelated_files(outfolder,pix,apparent_magnitude_flag,grouping,nside_mthr,mthr_percentile=mthr_percentile)\n')
    fp.write('for pix in tqdm(filled_pixels,desc=\'Calculating redshift grid\'):\n')
    fp.write('\ticarogw.catalog.get_redshift_grid_for_files(outfolder,pix,grouping,cosmo_ref,Nintegration=Nintegration,Numsigma=Numsigma,zcut=zcut)\n')

    f = open(os.path.join(home_folder,'calc_mthr_and_grid.sh'),'w')
    f.write('#!/bin/bash')
    f.write('\n')
    f.write('MYJOB_DIR='+home_folder)
    f.write('\n')
    f.write('cd ${MYJOB_DIR}')
    f.write('\n')
    f.write('python calc_mthr_and_grid.py $1 $2')
    f.close()

    f = open(os.path.join(home_folder,'calc_mthr_and_grid.sub'),'w')
    f.write('universe = vanilla\n')
    f.write('getenv = True\n')
    f.write(f'executable = {home_folder}/calc_mthr_and_grid.sh \n')
    f.write('accounting_group = '+agroup+'\n')
    f.write('accounting_group_user = '+uname)
    f.write('\n')
    f.write('request_memory = 8G \n')
    f.write('request_cpus = 1 \n')
    f.write('request_disk = 10G \n')    
    f.write('arguments = $(Item) $(Item2) \n')

    f.write(f'output = {home_folder}/logs/calc_mthr_and_grid_$(Item)_$(Item2).stdout \n')
    f.write(f'error = {home_folder}/logs/calc_mthr_and_grid_$(Item)_$(Item2).stderr \n')
    f.write('Requirements = TARGET.Dual =!= True \n')
    f.write('queue Item, Item2 from {:s}\n'.format(os.path.join(home_folder,'queue_NaN.txt')))
    f.close()
    
    
    os.system('chmod a+x '+os.path.join(home_folder,'*.sh'))    


def write_condor_files_initialize_icarogw_catalog(home_folder,outfolder, outfile,grouping ,uname='simone.mastrogiovanni', agroup='ligo.dev.o4.cbc.hubble.icarogw'):
    '''
    Writes the python scripts and condor files to create the icarogw file

    Parameters
    ----------
    home_folder: str
        Where to place the running scripts
    outfolder: str
        Where to place the pixelated catalog
    outfile: str
        Name to give to the icarogw file
    grouping: str
        Name of the new group to create in the pixelated files to save the apparent mthr cut
    uname: str
        Username for condor
    agroup: str
        Accounting group for condor
    '''
    
    fp = open(os.path.join(home_folder,'initialize_catalog.py'),'w')
    fp.write('import icarogw \n')
    fp.write('outfolder = \'{:s}\' \n'.format(outfolder))
    fp.write('outfile = \'{:s}\' \n'.format(outfile))
    fp.write('grouping = \'{:s}\' \n'.format(grouping))
    fp.write('icarogw.catalog.initialize_icarogw_catalog(outfolder,outfile,grouping)\n')

    f = open(os.path.join(home_folder,'initialize_catalog.sh'),'w')
    f.write('#!/bin/bash')
    f.write('\n')
    f.write('MYJOB_DIR='+home_folder)
    f.write('\n')
    f.write('cd ${MYJOB_DIR}')
    f.write('\n')
    f.write('python initialize_catalog.py')
    f.close()

    f = open(os.path.join(home_folder,'initialize_catalog.sub'),'w')
    f.write('universe = vanilla\n')
    f.write('getenv = True\n')
    f.write(f'executable = {home_folder}/initialize_catalog.sh \n')
    f.write('accounting_group = '+agroup+'\n')
    f.write('accounting_group_user = '+uname)
    f.write('\n')
    f.write('request_memory = 4G \n')
    f.write('request_cpus = 1 \n')
    f.write('request_disk = 4G \n')    
    f.write(f'output = {home_folder}/logs/initialize_catalog.stdout \n')
    f.write(f'error = {home_folder}/logs/initialize_catalog.stderr \n')
    f.write('Requirements = TARGET.Dual =!= True \n')
    f.write('queue\n')
    f.close()

    os.system('chmod a+x '+os.path.join(home_folder,'*.sh'))    

def write_condor_files_calculate_interpolant(home_folder,outfolder,grouping, subgrouping,band,epsilon,NumJobs,ptype='gaussian',uname='simone.mastrogiovanni',
agroup='ligo.dev.o4.cbc.hubble.icarogw'):
    '''
    Writes the python scripts and condor files to removes NaNs and calculate apparent magnitude threshold cut for galaxy catalog

    Parameters
    ----------
    home_folder: str
        Where to place the running scripts
    outfolder: str
        Where to place the pixelated catalog
    grouping: str
        Name of the new group to create in the pixelated files to save the apparent mthr cut
    subgrouping: str
        Name of the new subgroup for the galaxy catalog interpolant
    band: str
        icarogw EM band for the Schecter function
    epsilon: float
        Exponent of the luminosity weight, e.g. epsilon=1 is p(L) propto L
    NumJobs: int
        Number of jubs to run on condor
    ptype: str
        Type of EM likelihood, default is gaussian.
    uname: str
        Username for condor
    agroup: str
        Accounting group for condor
    '''

    filled_pixels = np.genfromtxt(os.path.join(outfolder,'filled_pixels.txt')).astype(int)
    NumPix = int(np.ceil(len(filled_pixels)/NumJobs))

    bot = np.arange(0,len(filled_pixels),NumPix)
    top = np.arange(NumPix,len(filled_pixels)+NumPix,NumPix)
    np.savetxt(os.path.join(home_folder,'queue_interpolant.txt'),np.column_stack([bot,top]),fmt='%d')

    fp = open(os.path.join(home_folder,'calc_interpolant.py'),'w')
    fp.write('import icarogw \n')
    fp.write('import numpy as np \n')
    fp.write('from tqdm import tqdm \n')
    fp.write('import sys \n')

    fp.write('from astropy.cosmology import Planck15 \n')
    fp.write('cosmo_ref = icarogw.cosmology.astropycosmology(zmax=10.)\n')
    fp.write('cosmo_ref.build_cosmology(Planck15)\n')
    
    fp.write('outfolder = \'{:s}\' \n'.format(outfolder))
    fp.write('grouping = \'{:s}\' \n'.format(grouping))
    fp.write('subgrouping = \'{:s}\' \n'.format(subgrouping))
    fp.write('band = \'{:s}\' \n'.format(band))
    fp.write('epsilon = {:f} \n'.format(epsilon))
    fp.write('ptype = \'{:s}\' \n'.format(ptype))

    
    fp.write('bot_pix = int(sys.argv[1])\n')
    fp.write('top_pix = int(sys.argv[2])\n')
    fp.write('filled_pixels = np.genfromtxt(\'{:s}\').astype(int) \n'.format(
        os.path.join(outfolder,'filled_pixels.txt')))
    fp.write('z_grid = np.genfromtxt(\'{:s}\') \n'.format(
        os.path.join(outfolder,'{:s}_common_zgrid.txt'.format(grouping))))
    fp.write('filled_pixels = filled_pixels[bot_pix:top_pix] \n')
    fp.write('for pix in tqdm(filled_pixels,desc=\'Calculating interpolant\'):\n')
    fp.write('\ticarogw.catalog.calculate_interpolant_files(outfolder,z_grid,pix,grouping,subgrouping,band,cosmo_ref,epsilon,ptype=ptype)\n')
    fp.close()

    f = open(os.path.join(home_folder,'calc_interpolant.sh'),'w')
    f.write('#!/bin/bash')
    f.write('\n')
    f.write('MYJOB_DIR='+home_folder)
    f.write('\n')
    f.write('cd ${MYJOB_DIR}')
    f.write('\n')
    f.write('python calc_interpolant.py $1 $2')
    f.close()

    f = open(os.path.join(home_folder,'calc_interpolant.sub'),'w')
    f.write('universe = vanilla\n')
    f.write('getenv = True\n')
    f.write(f'executable = {home_folder}/calc_interpolant.sh \n')
    f.write('accounting_group = '+agroup+'\n')
    f.write('accounting_group_user = '+uname)
    f.write('\n')
    f.write('request_memory = 8G \n')
    f.write('request_cpus = 1 \n')
    f.write('request_disk = 10G \n')    
    f.write('arguments = $(Item) $(Item2) \n')

    f.write(f'output = {home_folder}/logs/calc_interpolant_$(Item)_$(Item2).stdout \n')
    f.write(f'error = {home_folder}/logs/calc_interpolant_$(Item)_$(Item2).stderr \n')
    f.write('Requirements = TARGET.Dual =!= True \n')
    f.write('queue Item, Item2 from {:s}\n'.format(os.path.join(home_folder,'queue_interpolant.txt')))
    f.close()

    
    os.system('chmod a+x '+os.path.join(home_folder,'*.sh'))   

def write_condor_files_finish_catalog(home_folder,outfolder, outfile,grouping, subgrouping, uname='simone.mastrogiovanni', agroup='ligo.dev.o4.cbc.hubble.icarogw'):
    '''
    Writes the python scripts to finish the icarogw catalog given the files with interpolants
    
    Parameters
    ----------
    home_folder: str
        Where to place the running scripts
    outfolder: str
        Where to place the pixelated catalog
    outfile: str
        Name to give to the icarogw file
    grouping: str
        Name of the new group to create in the pixelated files to save the apparent mthr cut
    subgrouping: str
        Name of the new subgroup for the galaxy catalog interpolant
    uname: str
        Username for condor
    agroup: str
        Accounting group for condor
    '''
    
    fp = open(os.path.join(home_folder,'finish_catalog.py'),'w')
    fp.write('import icarogw \n')
    fp.write('outfolder = \'{:s}\' \n'.format(outfolder))
    fp.write('outfile = \'{:s}\' \n'.format(outfile))
    fp.write('grouping = \'{:s}\' \n'.format(grouping))
    fp.write('subgrouping = \'{:s}\' \n'.format(subgrouping))
    fp.write('icat = icarogw.catalog.icarogw_catalog(outfile,grouping,subgrouping)\n')
    fp.write('icat.build_from_pixelated_files(outfolder)\n')
    fp.write('icat.save_to_hdf5_file()\n')
 
    f = open(os.path.join(home_folder,'finish_catalog.sh'),'w')
    f.write('#!/bin/bash')
    f.write('\n')
    f.write('MYJOB_DIR='+home_folder)
    f.write('\n')
    f.write('cd ${MYJOB_DIR}')
    f.write('\n')
    f.write('python finish_catalog.py')
    f.close()

    f = open(os.path.join(home_folder,'finish_catalog.sub'),'w')
    f.write('universe = vanilla\n')
    f.write('getenv = True\n')
    f.write(f'executable = {home_folder}/finish_catalog.sh \n')
    f.write('accounting_group = '+agroup+'\n')
    f.write('accounting_group_user = '+uname)
    f.write('\n')
    f.write('request_memory = 4G \n')
    f.write('request_cpus = 1 \n')
    f.write('request_disk = 4G \n')    
    f.write(f'output = {home_folder}/logs/finish_catalog.stdout \n')
    f.write(f'error = {home_folder}/logs/finish_catalog.stderr \n')
    f.write('Requirements = TARGET.Dual =!= True \n')
    f.write('queue\n')
    f.close()

    os.system('chmod a+x '+os.path.join(home_folder,'*.sh'))    

import os
import numpy as np

def write_all_scripts_catalog(
    home_folder, outfolder, nside, fields_to_take, grouping, apparent_magnitude_flag,
    nside_mthr, mthr_percentile, Nintegration, Numsigma, zcut, outfile, subgrouping,
    band, epsilon, NumJobs, uname='albert.einstein', agroup='ligo.dev.o4.cbc.hubble.icarogw'
):
    '''
    Write all scripts and condor files needed to build the galaxy catalog, 
    and create a DAG file for producing the catalog on condor.

    Parameters
    ----------
    home_folder : str   | Path to place running scripts
    outfolder   : str   | Path to place pixelated catalog
    nside       : int   | Pixelization parameter
    fields_to_take: list| Groups to take from pixelated files
    grouping    : str   | Name of new group for mthr cut
    apparent_magnitude_flag: str | Group of m to use for mthr
    nside_mthr  : int   | Nside for apparent mag threshold
    mthr_percentile: float| Percentile (0-100) for mthr
    Nintegration: int/np.ndarray | Number of int points or grid
    Numsigma    : int   | Sigmas for EM likelihood
    zcut        : float | Redshift cutoff
    outfile     : str   | Name of icarogw file
    subgrouping : str   | New subgroup for catalog interpolant
    band        : str   | EM band for Schechter function
    epsilon     : float | Luminosity weight exponent
    NumJobs     : int   | Number of condor jobs
    uname       : str   | Condor username
    agroup      : str   | Condor accounting group
    '''

    write_condor_files_catalog(
        home_folder=home_folder, outfolder=outfolder, nside=nside, uname=uname, agroup=agroup
    )

    with open(os.path.join(home_folder, 'get_scripts.py'), 'w') as fp:
        fp.write('import icarogw\n')
        fp.write('import numpy as np\n')
        fp.write(f"home_folder = '{home_folder}'\n")
        fp.write(f"outfolder = '{outfolder}'\n")
        # fields_to_take
        fp.write('fields_to_take = [{}]\n'.format(
            ', '.join(f"'{f}'" for f in fields_to_take)
        ))
        fp.write(f"grouping = '{grouping}'\n")
        fp.write(f"apparent_magnitude_flag = '{apparent_magnitude_flag}'\n")
        fp.write(f"nside_mthr = {nside_mthr}\n")
        fp.write(f"mthr_percentile = {mthr_percentile:.6f}\n")
        if isinstance(Nintegration, np.ndarray):
            fp.write(
                "Nintegration = np.logspace(np.log10({:.6f}), np.log10({:.6f}), {})\n"
                .format(Nintegration.min(), Nintegration.max(), len(Nintegration))
            )
        else:
            fp.write(f"Nintegration = {Nintegration}\n")
        fp.write(f"Numsigma = {Numsigma}\n")
        fp.write(f"zcut = {zcut:.6f}\n")
        fp.write(f"outfile = '{outfile}'\n")
        fp.write(f"subgrouping = '{subgrouping}'\n")
        fp.write(f"band = '{band}'\n")
        fp.write(f"epsilon = {epsilon:.6f}\n")
        fp.write(f"NumJobs = {NumJobs}\n\n")
        fp.write(f"uname = '{uname}'\n")
        fp.write(f"agroup = '{agroup}'\n\n")

        fp.write(
            "# Write the condor files for NaNs and mthr computation\n"
            "icarogw.utils.write_condor_files_nan_removal_mthr_computation(\n"
            "    home_folder=home_folder,\n"
            "    outfolder=outfolder,\n"
            "    fields_to_take=fields_to_take,\n"
            "    grouping=grouping,\n"
            "    apparent_magnitude_flag=apparent_magnitude_flag,\n"
            "    nside_mthr=nside_mthr,\n"
            "    mthr_percentile=mthr_percentile,\n"
            "    Nintegration=Nintegration,\n"
            "    Numsigma=Numsigma,\n"
            "    zcut=zcut,\n"
            "    NumJobs=NumJobs,\n"
            "    uname=uname,\n"
            "    agroup=agroup\n"
            ")\n\n"
        )
        fp.write(
            "# Initialize the icarogw file\n"
            "icarogw.utils.write_condor_files_initialize_icarogw_catalog(\n"
            "    home_folder=home_folder,\n"
            "    outfolder=outfolder,\n"
            "    outfile=outfile,\n"
            "    grouping=grouping,\n"
            "    uname=uname,\n"
            "    agroup=agroup\n"
            ")\n\n"
        )
        fp.write(
            "# Calculate the interpolant\n"
            "icarogw.utils.write_condor_files_calculate_interpolant(\n"
            "    home_folder=home_folder,\n"
            "    outfolder=outfolder,\n"
            "    grouping=grouping,\n"
            "    subgrouping=subgrouping,\n"
            "    band=band,\n"
            "    epsilon=epsilon,\n"
            "    NumJobs=NumJobs,\n"
            "    uname=uname,\n"
            "    agroup=agroup\n"
            ")\n\n"
        )
        fp.write(
            "# Finish catalog\n"
            "icarogw.utils.write_condor_files_finish_catalog(\n"
            "    home_folder=home_folder,\n"
            "    outfolder=outfolder,\n"
            "    outfile=outfile,\n"
            "    grouping=grouping,\n"
            "    subgrouping=subgrouping,\n"
            "    uname=uname,\n"
            "    agroup=agroup\n"
            ")\n"
        )

    with open(os.path.join(home_folder, 'get_scripts.sh'), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f"MYJOB_DIR={home_folder}\n")
        f.write("cd ${MYJOB_DIR}\n")
        f.write("python get_scripts.py\n")

    with open(os.path.join(home_folder, 'get_scripts.sub'), 'w') as f:
        f.write(
            "universe = vanilla\n"
            "getenv = True\n"
            f"executable = {home_folder}/get_scripts.sh\n"
            f"accounting_group = {agroup}\n"
            f"accounting_group_user = {uname}\n"
            "request_memory = 4G\n"
            "request_cpus = 1\n"
            "request_disk = 4G\n"
            f"output = {home_folder}/logs/get_scripts.stdout\n"
            f"error = {home_folder}/logs/get_scripts.stderr\n"
            "Requirements = TARGET.Dual =!= True\n"
            "queue\n"
        )

    with open(os.path.join(home_folder, 'produce_cat.dag'), 'w') as fp:
        fp.write(
            f"JOB A {home_folder}/make_pixel_files.sub\n"
            f"JOB A1 {home_folder}/get_scripts.sub\n"
            f"JOB B {home_folder}/clear_NaNs.sub\n"
            f"JOB C {home_folder}/calc_mthr_and_grid.sub\n"
            f"JOB D {home_folder}/initialize_catalog.sub\n"
            f"JOB E {home_folder}/calc_interpolant.sub\n"
            f"JOB F {home_folder}/finish_catalog.sub\n\n"
            "PARENT A CHILD A1\n"
            "PARENT A1 CHILD B\n"
            "PARENT B CHILD C\n"
            "PARENT C CHILD D\n"
            "PARENT D CHILD E\n"
            "PARENT E CHILD F\n"
        )

    os.system(f'chmod a+x {os.path.join(home_folder, "*.sh")}')