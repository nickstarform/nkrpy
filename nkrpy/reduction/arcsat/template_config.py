"""Configuration file for arcsat data reduction."""
work = ''  # Leave blank if current directory, otherwise provide full path
flats = './flat_dir'  # Directory of flats
darks = './dark_dir'  # Directory of Darks
bias = './bias_dir'  # Directory of bias
science = './science_dir'  # directory of science targets
createplot = False  # Create plots (pngs)?
createfits = False  # Create fits (files)?
# if both createplot and createfits are false, code won't do anything
calibration = './cals'  # directory to store calibration frames
destination = './finals'  # destination directory for data reduction
color = False  # Try to create color plots?

# end of file
