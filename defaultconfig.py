#!/usr/bin/env python
'''
Name  : Default Config, defaultconfig.py
Author: Nickalas Reynolds
Date  : Fall 2017
Misc  : Default Configuration file to help the user determine input
        parameters
Notes : DO NOT CHANGE THE STYLE OF THE FILE
'''

config = {
'current'   : '',  # name of this input file
'originput' : '',  # original input file
'pickle'    : '',  # pickle file name
'source'    : '',  # source name
'order'     : [[3,'2.05~2.15','2.25~2.35','include'],
               [4,'1.6~1.75','1.5~1.55','include']
              ], 
                                         # 2d list of [order,lower~upper,type] where type is either 
                                         # include/exclude where the lower~upper range is either the 
                                         # included range or excluded range
'rawrange'  : ['0.94~2.5','0.5e-15~1.5e-15'], # raw xrange and yrange
'flatrange'  : ['0.94~2.5','0.5e-15~1.5e-15'], # raw xrange and yrange
'lines'     : [r'Br $\gamma$',r'Pa $\beta$'], 
'input'     : 0.8,
}
###############################################################################
##########################DO NOT CHANGE STYLE OF FILE##########################
###############################################################################
