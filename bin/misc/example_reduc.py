"""
This script was written for CASA 5.1.1

Datasets calibrated (in order of date observed):
SB1: 2013.1.00498.S
     Observed 21 July 2015 (1 execution block)
LB1: 2016.1.00484.L
     Observed 07 September 2017 and 03 October 2017 (2 execution blocks)

reducer: J. Huang
"""

""" Starting matter """
import os
execfile('reduction_utils.py')

""" Input for loading data """
prefix  = 'Elias27'
SB1_path = '/full_path/to_calibrated/msfile.ms'
LB1_path = '/full_path/to_calibrated/msfile.ms'

""" Some preliminary flagging """
flagdata(vis=SB1_path, spw='1:167~215,5:49~72,6:806~829', field='Elia_2-27')


# Note that if you are downloading data from the archive, your SPW numbering 
# may differ from this script, depending on how you split your data out!
data_params = {'SB1': {'vis' : SB1_path,
                       'name' : 'SB1',
                       'field': 'Elia_2-27',
                       'line_spws': np.array([1,5,6]), # CO 13CO, C18O SPWs 
                       'line_freqs': np.array([2.30538e11, 2.2039868420e11,
                                               2.1956035410e11]),
                      },
               'LB1': {'vis' : LB1_path,
                       'name' : 'LB1',
                       'field' : 'Elias_27',
                       'line_spws': np.array([3,7]), # CO SPWs
                       'line_freqs': np.array([2.30538e11, 2.30538e11]),
                      }
               }



""" Split out the SB data, with the continuum SPWs averaged down """
SB_msfile = data_params['SB1']['vis']
if os.path.isdir(SB_msfile+'.flagversions/flags.before_cont_flags'):
    flagmanager(vis=SB_msfile, mode='delete', versionname='before_cont_flags')
flagmanager(vis=SB_msfile, mode='save', versionname='before_cont_flags',
            comment='Flag states before spectral lines are flagged')
flagdata(vis=SB_msfile, mode='manual', spw='5:34~94, 6:791~851',
         flagbackup=False, field=data_params['SB1']['field'])
os.system('rm -rf %s*' % prefix+'_SB1_lines_exec0')
split(vis=data_params['SB1']['vis'], field=data_params['SB1']['field'],
      spw='0~7', outputvis=prefix+'_SB1_lines_exec0.ms',
      width=[120,1,128,60,60,960,960,128], datacolumn='data',
      intent='OBSERVE_TARGET#ON_SOURCE', keepflags=False)
flagmanager(vis=SB_msfile, mode='restore', versionname='before_cont_flags')

""" Some additional flagging on SB data """
flagmanager(vis=prefix+'_SB1_lines_exec0.ms', mode='save',
            versionname='init_cal_flags',
            comment='Flag states immediately after initial calibration')
flagdata(vis=prefix+'_SB1_lines_exec0.ms', mode='manual', spw='1,3,5',
         flagbackup=False, field=data_params['SB1']['field'], scan='32',
         antenna='DA46')
flagdata(vis=prefix+'_SB1_lines_exec0.ms', mode='manual', spw='3,4,6',
         flagbackup=False, field=data_params['SB1']['field'], scan='18,32,37',
         antenna='DA59')
flagdata(vis=prefix+'_SB1_lines_exec0.ms', mode='manual', spw='3,4',
         flagbackup=False, field=data_params['SB1']['field'], scan='13,18',
         antenna='DV06')
flagdata(vis=prefix+'_SB1_lines_exec0.ms', mode='manual', spw='3,4',
         flagbackup=False, field=data_params['SB1']['field'], scan='32',
         antenna='DV08')
flagdata(vis=prefix+'_SB1_lines_exec0.ms', mode='manual', spw='3',
         flagbackup=False, field=data_params['SB1']['field'], scan='32,37',
         antenna='DV18')


""" Split out the LB data, with the continuum SPWs averaged down """
os.system('rm -rf '+prefix+'_LB1_lines_exec0*')
split(vis=data_params['LB1']['vis'], field=data_params['LB1']['field'],
      spw='0~3', outputvis=prefix+'_LB1_lines_exec0.ms', width=[128,128,128,1],
      timebin='6s', datacolumn='data', intent='OBSERVE_TARGET#ON_SOURCE',
      keepflags=False)
os.system('rm -rf '+prefix+'_LB1_lines_exec1*')
split(vis=data_params['LB1']['vis'], field=data_params['LB1']['field'],
      spw='4~7', outputvis=prefix+'_LB1_lines_exec1.ms', width=[128,128,128,1],
      timebin='6s', datacolumn='data', intent='OBSERVE_TARGET#ON_SOURCE',
      keepflags=False)



""" Apply same shifts and re-scalings as for continuum """
common_dir = 'J2000 16h26m45.022s -024.23.08.273'
shiftname = prefix+'_SB1_lines_exec0_shift'
os.system('rm -rf %s.ms' % shiftname)
fixvis(vis=prefix+'_SB1_lines_exec0.ms', outputvis=shiftname+'.ms',
       field=data_params['SB1']['field'],
       phasecenter='J2000 16h26m45.021955s -24d23m08.25057s')
fixplanets(vis=shiftname+'.ms', field=data_params['SB1']['field'],
           direction=common_dir)
shiftname = prefix+'_LB1_exec0_shift'
os.system('rm -rf %s.ms' % shiftname)
fixvis(vis=prefix+'_LB1_exec0.ms', outputvis=shiftname+'.ms',
       field=data_params['LB1']['field'],
       phasecenter='ICRS 16h26m45.021309s -24d23m08.28623s')
fixplanets(vis=shiftname+'.ms', field=data_params['LB1']['field'],
           direction=common_dir)
shiftname = prefix+'_LB1_exec1_shift'
os.system('rm -rf %s.ms' % shiftname)
fixvis(vis=prefix+'_LB1_exec1.ms', outputvis=shiftname+'.ms',
       field=data_params['LB1']['field'],
       phasecenter='ICRS 16h26m45.021309s -24d23m08.28623s')
fixplanets(vis=shiftname+'.ms', field=data_params['LB1']['field'],
           direction=common_dir)



""" Apply the self-calibration solutions to the SB data """
SB_CO = prefix+'_SB_CO'
os.system('rm -rf %s*' % SB_CO)
os.system('cp -r '+prefix+'_SB1_lines_exec0_shift.ms '+SB_CO+'.ms')

applycal(vis=SB_CO+'.ms', spw='0~7',
         gaintable=[prefix+'_SB.p1', prefix+'_SB.p2', prefix+'_SB.ap'],
         interp='linearPD', calwt=True, flagbackup=False)

os.system('rm -rf '+SB_CO+'_selfcal.ms*')
split(vis=SB_CO+'.ms', outputvis=SB_CO+'_selfcal.ms', datacolumn='corrected',
      keepflags=False)

# save some space by deleting intermediate files
os.system('rm -rf '+SB_CO+'.ms*')
os.system('rm -rf '+prefix+'_SB1_lines_exec*ms*')



""" Now apply the self-calibration solutions to the combined dataset """
combined_CO = prefix+'_combined_CO'
os.system('rm -rf '+combined_CO+'.ms*')
concat(vis=[SB_CO+'_selfcal.ms',
            prefix+'_LB1_lines_exec0_shift.ms', 
            prefix+'_LB1_lines_exec1_shift.ms'],
       concatvis=combined_CO+'.ms', dirtol='0.1arcsec', copypointing=False)

applycal(vis=combined_CO+'.ms', spw='0~15',
         spwmap=[[0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12]]*5,
         gaintable=[prefix+'_combined.p1', prefix+'_combined.p2',
                    prefix+'_combined.p3', prefix+'_combined.p4', 
                    prefix+'_combined.ap'],
         interp='linearPD', calwt=True, applymode='calonly')

os.system('rm -rf '+combined_CO+'_selfcal.ms*')
split(vis=combined_CO+'.ms', outputvis=combined_CO+'_selfcal.ms',
      field='', spw='1,11,15', datacolumn='corrected', keepflags=False)

# save some space by deleting intermediate files
os.system('rm -rf '+combined_CO+'.ms*')
os.system('rm -rf '+prefix+'_LB1_lines_exec*ms*')



""" Continuum subtraction """
fitspw="0:0~167;215~1920, 1:0~1880;1950~3839, 2:0~1880;1950~3839"
os.system('rm -rf '+combined_CO+'_selfcal.ms.contsub*')
uvcontsub(vis=combined_CO+'_selfcal.ms', spw='0~2', fitspw=fitspw,
          excludechans=False, solint='int', fitorder=1, want_cont=False)



""" Define the channels of interest """
chanstart = '-8.8km/s'
chanwidth = '0.35km/s'
nchan = 60



""" Split and regrid into the channels of interest """
# Continuum-subtracted
SB_only = prefix+'_CO_SBonly.ms.contsub'
os.system('rm -rf '+SB_only+'*')
split(vis=combined_CO+'_selfcal.ms.contsub', outputvis=SB_only, spw='0',
      datacolumn='data')

SB_cvel = SB_only+'.cvel'
os.system('rm -rf '+SB_cvel+'*')
mstransform(vis=SB_only, outputvis=SB_cvel, keepflags=False, datacolumn='data',
            regridms=True, mode='velocity', start=chanstart, width=chanwidth,
            nchan=nchan, outframe='LSRK', veltype='radio',
            restfreq='230.538GHz')

LB_only = prefix+'_CO_LBonly.ms.contsub'
os.system('rm -rf '+LB_only+'*')
split(vis=combined_CO+'_selfcal.ms.contsub', outputvis=LB_only, spw='1,2',
      datacolumn='data')

LB_cvel = LB_only+'.cvel'
os.system('rm -rf '+LB_cvel+'*')
mstransform(vis=LB_only, outputvis=LB_cvel, keepflags=False, datacolumn='data',
            regridms=True, mode='velocity', start=chanstart, width=chanwidth,
            nchan=nchan, outframe='LSRK', veltype='radio',
            restfreq='230.538GHz')

contsub_concat = prefix+'_COcube.ms.contsub'
os.system('rm -rf '+contsub_concat)
concat(vis=[SB_cvel, LB_cvel], concatvis=contsub_concat, dirtol='0.1arcsec',
       copypointing=False)

# With continuum
cSB_only = prefix+'_CO_SBonly.ms'
os.system('rm -rf '+cSB_only)
split(vis=combined_CO+'_selfcal.ms', outputvis=cSB_only, spw='0',
      datacolumn='data')

cSB_cvel = cSB_only+'.cvel'
os.system('rm -rf '+cSB_cvel)
mstransform(vis=cSB_only, outputvis=cSB_cvel, keepflags=False,
            datacolumn='data', regridms=True, mode='velocity', start=chanstart,
            width=chanwidth, nchan=nchan, outframe='LSRK', veltype='radio',
            restfreq='230.538GHz')

cLB_only = prefix+'_CO_LBonly.ms'
os.system('rm -rf '+cLB_only)
split(vis=combined_CO+'_selfcal.ms', outputvis=cLB_only, spw='1,2',
      datacolumn='data')

cLB_cvel = cLB_only+'.cvel'
os.system('rm -rf '+cLB_cvel)
mstransform(vis=cLB_only, outputvis=cLB_cvel, keepflags=False,
            datacolumn='data', regridms=True, mode='velocity', start=chanstart,
            width=chanwidth, nchan=nchan, outframe='LSRK', veltype='radio',
            restfreq='230.538GHz')

cont_concat = prefix+'_COcube.ms'
os.system('rm -rf '+cont_concat)
concat(vis=[cSB_cvel, cLB_cvel], concatvis=cont_concat, dirtol='0.1arcsec',
       copypointing=False)



""" Imaging (continuum-subtracted only) """
imagename = prefix+'_CO'
for ext in ['.image','.mask','.model','.pb','.psf','.residual','.sumwt']:
    os.system('rm -rf '+ imagename + ext)
tclean(vis=prefix+'_COcube.ms.contsub',
       imagename=imagename, specmode='cube', imsize=1500,
       deconvolver='multiscale', start=chanstart, width=chanwidth, nchan=nchan,
       outframe='LSRK', veltype='radio', restfreq='230.538GHz',
       cell='0.01arcsec', scales = [0,10,30,100,200,300], gain=0.1, niter=50000,
       weighting='briggs', robust=1.0, threshold='5mJy', 
       uvtaper=['.1arcsec', '0.07arcsec', '-35deg'], uvrange='>20klambda',
       interactive=True, nterms=1, restoringbeam='common')
# Note that masking is done interactively



"""
Final outputs
"""

""" Save the final MS files """
os.system('tar cvzf '+prefix+'_CO.ms.tgz '+prefix+'_COcube.ms.contsub')
os.system('tar cvzf '+prefix+'_COcont.ms.tgz '+prefix+'_COcube.ms')

""" Save the imaged datacube """
exportfits(imagename+'.image', imagename+'.fits')
