"""."""
# flake8: noqa
# cython modules

# internal modules
import importlib
import os
from sys import version
import re
import pickle
import glob

# external modules
import aplpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython import embed
from colormap import mainColorMap

# relative modules
from ....publication.plots import set_style
from .... import typecheck
from .... import convert
from ....io import config
# global attributes
__all__ = ('test', 'main')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__cwd__ = os.getcwd()
set_style()
icrs2deg, deg2icrs = convert.icrs2deg, convert.deg2icrs

assert aplpy.__version__ == '1.1.1'

base_config_types = {
    # input file params
    'load': {'dtype': bool, 'opt': False},
    'load_dir': {'dtype': str, 'opt': True},
    'cont_filename': {'dtype': str, 'opt': True},
    'mom_filename': {'dtype': str, 'opt': True},
    'cont_image': {'dtype': str, 'opt': True},
    'mom_image': {'dtype': str, 'opt': True},
    # save params
    'save': {'dtype': bool, 'opt': False},
    'save_dir': {'dtype': str, 'opt': True},
    'save_prefix': {'dtype': str, 'opt': True},
    'save_suffix': {'dtype': list, 'opt': True},
    'tightplot': {'dtype': bool, 'opt': True},
    # geom config params
    'ra': {'dtype': float, 'opt': False},
    'dec': {'dtype': float, 'opt': False},
    'distance': {'dtype': float, 'opt': False},
    # gen plot params
    'ax': {'dtype': mpl.axes._subplots.AxesSubplot, 'opt': True},
    'fig': {'dtype': mpl.figure.Figure, 'opt': True},
    'beam_show': {'dtype': bool, 'opt': False},
    'beam_pad': {'dtype': int, 'opt': False},
    'scalebar_size': {'dtype': float, 'opt': True},
    'scalebar_show': {'dtype': bool, 'opt': False},
    'imagestretch': {'dtype': str, 'opt': True},
    'label_colour': {'dtype': str, 'opt': False},
    'overlay_tex_colour': {'dtype': str, 'opt': False},
    'figuresize': {'dtype': list, 'opt': True},
    'imagesize': {'dtype': list, 'opt': False},
    'colorbar_show': {'dtype': bool, 'opt': False},
    'title': {'dtype': str, 'opt': True},
    'legend_show': {'dtype': bool, 'opt': False},
    'axis_label_show': {'dtype': bool, 'opt': False},
    'axis_ticks_show': {'dtype': bool, 'opt': False},
    # cont plot params
    'grayscale': {'dtype': bool, 'opt': False},
    'colormap': {'dtype': str, 'opt': True},
    'cont_rms': {'dtype': float, 'opt': True},
    'cont_vmin': {'dtype': int, 'opt': True},
    'cont_vmax': {'dtype': int, 'opt': True},
    'cont_contour_show': {'dtype': bool, 'opt': False},
    'cont_allow_neg_contour': {'dtype': bool, 'opt': False},
    'cont_contour_start': {'dtype': int, 'opt': True},
    'cont_contour_iter': {'dtype': int, 'opt': True},
    'cont_contour_end': {'dtype': int, 'opt': True},
    # mom plot params
    'mom_allow_neg_contour': {'dtype': bool, 'opt': False},
    'moment_params': {'dtype': dict, 'opt': True}, # {red_color:,blue_color:,red_start:, red_iter: red_end:, red_rms:,...}
    # marker
    'markers': {'dtype': dict, 'opt': True}, # {label: {marker_show:, marker_label_show:, color:,ra:,dec:,marker:,elipse:,xsize:,ysize:,angle:}}
    # outflow
    'outflows': {'dtype': dict, 'opt': True}, # {label: {ra:,dec:,x_offset:,y_offset:,show_red:, show_blue:, blue_angle:,red_angle:,length:}}
    # text
    'text': {'dtype': dict, 'opt': True}, # {label: {content, size, ra, dec, color}}
}
base_config_def = {
    
}



class mappingContours():
    """
    """
    def __init__(self,beam,cont,ra,dec,minpixval=0.,maxpixval=0.,\
                size=(1,1),scalebar=0.,distance=0.,name='',imagestretch='sqrt',\
                colororgray='false',contContourColor=False,cont_rms=1E-5,continuumContourParams=False,
                cont_contour=False,colormap='ds9',plotlabel='',misc='',\
                textcolor='',extension=('pdf',),showContourLabels=False,fontSize=20,\
                contourSize=1.,figSize=(5,5), titlepos=0.1, miscpos=0.2, labelpos=0.3, beam_pad=0, labelspace=10, customMarker=False, customMarkerSize=0, cfg={}):
        self.cont = f'{__cwd__}/{cont}'
        self.totalcfg = cfg
        self.ra = ra
        self.labelspace = labelspace
        self.dec = dec
        self.minpixval = minpixval * cont_rms
        self.maxpixval = maxpixval * cont_rms
        self.cont_rms = cont_rms
        self.continuumContourParams = continuumContourParams
        self.xsize,self.ysize = size
        #print("Size:",10,int(10*self.ysize/self.xsize))
        self.fig = mpl.figure(figsize=(figSize[0] ,figSize[1]))
        self.scalebar = scalebar
        self.distance = distance
        self.name = name
        self.imagestretch = imagestretch
        self.cont_contour = cont_contour
        self.contContourColor = contContourColor
        self.colororgray = colororgray
        if 'ds9' in colormap:
            mainColorMap('ds9')
        self.colormap = colormap
        self.plotlabel = plotlabel
        if textcolor == '':
            self.auto = True
            if colororgray == 'true':
                textcolor = 'white'
            else:
                textcolor = 'black'
        else:
            self.auto = False

        self.textcolor = textcolor
        self.extension = extension
        self.misc = misc
        self.showContourLabels = showContourLabels
        self.fontSize = fontSize
        self.contourSize = contourSize
        self.titlepos=titlepos
        self.miscpos=miscpos
        self.labelpos=labelpos
        self.beam_pad = beam_pad
        self.show_beam = beam
        self.customMarker = customMarker
        self.customMarkerPos = []
        self.customMarkerSize = customMarkerSize
        print('Cont image:',cont, f' with settings: {dir(self)}')

    def drawContinuum(self):
        """
        """
        if self.cont != "" :
            print('starting cont')
            self.gc1 = aplpy.FITSFigure(self.cont, figure=self.fig, dimensions=(0,1))
            if (self.colororgray.lower() == 'true'):
                self.gc1.show_colorscale(vmin=self.minpixval,vmax=self.maxpixval,
                    stretch=self.imagestretch,cmap=self.colormap) 
            if (self.colororgray.lower() == 'false'):
                self.gc1.show_grayscale(vmin=self.minpixval,vmax=self.maxpixval,
                    stretch=self.imagestretch,invert=True)
            self.setupContours(self.contContourColor,False, self.cont_contour,self.cont,False,
                               *self.continuumContourParams, self.cont_rms,'')
        print('finished cont')


    def addBeam(self,imageName):
        """
        """
        try:
            dele = mpl.figure(figsize=(1,1))
            self.ContBeam = aplpy.FITSFigure(imageName, figure=dele, dimensions=(0,1))
            self.ContBeam.add_beam()   
            dele = None
        except Exception as e:
            print('Cant add beam', e)
            pass

    def setupContinuum(self,setup,imageName):
        if setup and self.show_beam:
            self.addBeam(imageName)
            try:
                print('setup cont beam')
                self.gc1.add_beam()
            except Exception as e:
                print('Cant setup continuum',e)
                pass

    def setupContours(self,color,setup,show,imageName,neg,contourStart,contourInterval,contourNoise,label):
        """
        """
        if setup and self.show_beam:
            self.addBeam(imageName)
            try:
                print('setup contour beam')
                self.gc1.add_beam()
            except Exception as e:
                print('Cant setup Contour',e)
                pass
        if show:
            print('Showing contours')
            contours = self.drawContours(neg,contourStart,contourInterval,contourNoise)
            #print(contours)
            #print(imageName)
            ind = np.ravel(np.where(contours < 0))
            self.gc1.show_contour(imageName,levels=contours[ind],colors=color,linewidths=self.contourSize, linestyles='--')
            ind = np.ravel(np.where(contours > 0))
            self.gc1.show_contour(imageName,levels=contours[ind],colors=color,linewidths=self.contourSize)
            if self.showContourLabels:
                self.addLabel(color," contour: "+label)


    def drawContours(self,neg=False,contstart=None,continterval=None,contnoise=None):
        """
        """
        contours1=np.arange(contstart,contstart*10.0,continterval,dtype='float32')
        contours2=np.arange(contstart*10.0,contstart*40.0,continterval*3.0,dtype='float32')
        contours3=np.arange(contstart*40.0,contstart*100.0,continterval*10.0,dtype='float32')
        poscontours=np.concatenate((contours1,contours2,contours3))
        contours = poscontours * contnoise
        if neg:
            negcontours=poscontours[::-1]*(-1.0)
            contours=np.concatenate((poscontours,negcontours))*contnoise
            contours.sort()
        return contours

    def plotFormat(self):
        """
        """
        # centering
        width=self.xsize/3600.0
        height=self.ysize/3600.0
        print(f'Recentering: {self.ra},{self.dec},{width},{height}')
        #print(f'{self.gc1.image.get_size()}')
        self.gc1.recenter(x=self.ra,y=self.dec,width=width,height=height)
        print('recentered')

        # axis labels
        if showAxisLabels:
            print(f'showAxisLabels:')
            self.gc1.axis_labels.set_font(size=self.fontSize)
            self.gc1.tick_labels.set_style('colons')
            self.gc1.tick_labels.set_xformat('hh:mm:ss.s')
            self.gc1.tick_labels.set_yformat('dd:mm:ss.s')
            self.gc1.tick_labels.set_font(size=self.fontSize)
            self.gc1.axis_labels.set_ypad(-2*self.fontSize)
        else:
            print(f'!showAxisLabels:')
            self.gc1.axis_labels.hide()
            self.gc1.tick_labels.hide()

        if showAxisTicks:
            print(f'showAxisTicks:')
            self.gc1.ticks.set_color('black')
            self.gc1.ticks.show()
        else:
            print(f'!showAxisTicks:')
            self.gc1.ticks.hide()

        # scalebar
        if show_scalebar:
            print(f'show_scalebar:')
            self.gc1.add_scalebar(self.scalebar/3600.0,color=self.textcolor)
            self.gc1.scalebar.set_corner('bottom left')
            val = self.scalebar*self.distance   # to pc
            unit = 'au'
            if val > 2000:
                val = float(f'{val / (206266.30488698962):1.1e}')
                unit = 'pc'
            self.gc1.scalebar.set_label(str(self.scalebar)+\
                                        f'" ({val} {unit})')
            self.gc1.scalebar.set_linewidth(self.fontSize/3)
            self.gc1.scalebar.set_font_size(self.fontSize)

        # beam
        if self.show_beam:
            try:
                if typecheck(self.gc1.beam):
                    print('multiple beams')
                    for i, h in enumerate(self.gc1.beam):
                        if i == 0:
                            for x in self.ContBeam.beam._base_settings:
                                self.gc1.beam[i]._base_settings[x] = self.ContBeam.beam._base_settings[x]

                        self.gc1.beam[i].set_corner('bottom right')
                        self.gc1.beam[i].set_color(self.textcolor)
                        self.gc1.beam[i].set_hatch('+')
                        self.gc1.beam[i].set_alpha(1.0)
                        self.gc1.beam[i].set_pad(self.beam_pad * i)
                else:
                    print('single beam')
                    self.gc1.beam.set_corner('bottom right')
                    self.gc1.beam.set_color(self.textcolor)
                    self.gc1.beam.set_hatch('+')
                    self.gc1.beam.set_alpha(1.0)
            except:
                print('hit beam exception')
                try:
                    for x in self.ContBeam.beam._base_settings:
                        self.gc1.beam._base_settings[x] = self.ContBeam.beam._base_settings[x]
                except:
                    pass

        # title/misc labels
        fontConv = 3./0.04167 # pts/inche
        spaceReq = (self.fontSize/fontConv)/self.ysize # in % of y axis
        self.gc1.add_label(0.5, 1. - self.titlepos, r'{}'.format(self.name), \
            relative=True,size=self.fontSize,color=self.textcolor,weight='heavy')
        self.gc1.add_label(0.1, 1.-self.labelpos, r'{}'.format(self.plotlabel), \
            relative=True,size=self.fontSize-1,color=self.textcolor,weight='heavy')
        self.gc1.add_label(0.5, 1.-self.miscpos, r'{}'.format(self.misc), relative=True,\
            size=self.fontSize-2,color=self.textcolor,weight='heavy')
        if self.totalcfg.use_legend:
            if 'legend_pos' in self.totalcfg.__dict__:
                lp = self.totalcfg.legend_pos
            else:
                lp = 0
            self.gc1._ax1.legend(loc=lp)


    def drawOutflow(self,ra,dec,paR,paB):
        """draws outflows
        @input : xlength draws an outflow of this xlength
        @input : ylength "" ylength
        @input : ra places outflow at this pos
        @input : dec ""
        @input : paR and paB position angles of Red and blue outflow
        """
        xlength = self.xsize/8.
        ylength = self.ysize/8.
        if paR != None:
            dxred=xlength*np.cos(paR)
            dyred=ylength*np.sin(paR)
            self.gc1.show_arrows(ra,dec, dxred, dyred, color='red')
        if paB != None:
            dxblue=xlength*np.cos(paB)
            dyblue=ylength*np.sin(paB)
            self.gc1.show_arrows(ra,dec, dxblue, dyblue, color='blue')

    def showSources(self,ra,dec,sType: str='+',circle=False,size='0,0',color='white',label=''):
        """
        """
        #print(ra,dec,size)
        if sType.lower() == 'rectangle':
            w,h,a = map(float,size.split(','))
            if a == 0:
                self.gc1.show_rectangles(ra,dec,w/3600.,h/3600.,edgecolor=color,zorder=20, label=label, linewidth=self.contourSize)
            else:
                a *= np.pi / 180.
                left = ra + w/3600.
                right = ra - w/3600.
                up = dec + h/3600.
                down = dec - h/3600.
                newx = lambda x, y, theta: (x * np.cos(-theta) - y * np.sin(-theta)) + ra
                newy = lambda x, y, theta: (x * np.sin(-theta) + y * np.cos(-theta)) + dec
                ur = (newx(-w/3600, h/3600., a), newy(-w/3600, h/3600., a))
                ul = (newx(w/3600, h/3600., a), newy(w/3600, h/3600., a))
                lr = (newx(-w/3600, -h/3600., a), newy(-w/3600, -h/3600., a))
                ll = (newx(w/3600, -h/3600., a), newy(w/3600, -h/3600., a))
                from IPython import embed
                print(ra, dec, a, w, h, ur, ul, lr, ll)
                self.gc1.show_polygons([np.array([ur, ul, ll, lr]).T], color='red')
        elif circle.lower() == 'true':
            w,h,a = map(float,size.split(','))
            #print(ra,dec,w,h)
            self.gc1.show_ellipses(ra,dec,w/3600.,h/3600.,angle=a,edgecolor=color,zorder=20, linewidth=self.contourSize)
            self.gc1.show_markers(ra,dec, c=color,marker='_',zorder=-1, label=label)
        elif sType:
            try:
                size = int(size)
            except:
                size = 10*self.contourSize
            #print('gothere')
            self.gc1.show_markers(ra,dec, c=color,marker=sType,zorder=20, s=size, linewidths = self.totalcfg.customMarkerSize, label=label)
            #print('passed here')
        self.customMarkerPos.append([ra, dec])

    def showColorbar(self):
        """
        """
        self.gc1.add_colorbar()
        self.gc1.colorbar.set_width(0.15)
        self.gc1.colorbar.set_location('right')
        self.gc1.colorbar.set_axis_label_text('Jy/Beam')
        self.gc1.colorbar.set_axis_label_font(size=self.fontSize, weight='black', style='oblique')
        self.gc1.colorbar.set_font(size=self.fontSize, weight='bold')


    def addLabel(self,color,label):
        """
        """
        fontConv = 3./0.04167 # pts/inche
        spaceReq = (self.labelspace/fontConv)/self.ysize # in % of y axis
        if self.totalcfg.use_legend:
            return
        if label.strip(' ') != '':
            try:
                self.labelNum += spaceReq + 2./(fontConv * self.ysize )
                labelNum = self.labelNum
            except:
                labelNum = self.labelpos/(fontConv * self.ysize )
                self.labelNum = labelNum
            if self.customMarker:
                ra, dec = self.customMarkerPos[0]
                del self.customMarkerPos[0]
                self.gc1.add_label(ra - self.totalcfg.markerside * self.totalcfg.markersep/3600., dec, r'{}'.format(label), \
                    relative=False,size=self.fontSize,color=color)
            else:
                self.gc1.add_label(0.13, 1.-labelNum, r'{}'.format(label), \
                    relative=True,size=self.fontSize,color=color)

    def save(self,outfilename,dpi=300): 
        """plotting module
        @input extension tuple of extension types
        @input dpi, 400 is pretty high quality
        """ 
        self.gc1.list_layers()
        #self.fig.set_facecolor('black')
        # self.fig.tight_layout()
        for ext in self.extension:
            self.fig.savefig(outfilename+'.'+ext,dpi=dpi,format=ext,\
                facecolor=self.fig.get_facecolor(), edgecolor='none', bbox_inches = 'tight', pad_inches = 0)

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def main(configFname):
    '''Main calling function for the program
    @input : configFname is the name of the configuration file. Has a default value just incase
    Loads in all of the values found in the configuration file
    '''
    config = loadCfg(configFname)
    configback = config
    loadVariables(config)
    config = loadCfg('/'+__cpath__+'/colormap.py')
    loadVariables(config)
    config = configback

    if continuumOverlay[2].lower() == 'false':
        for molI,molV in enumerate(mols):
            for qualI1,qualV1 in enumerate(quals):
                ra,dec = map(float,parseRaDec(center[molI]).split(' '))
                if continuumOverlay[1] == 'both':
                    for colorOverlay in ('true','false'):
                        # setup plot
                        plot = mappingContours(show_beam,continuumOverlay[0],ra,dec,minpixval=continuumParams[0],\
                            maxpixval=continuumParams[1], size=imSize[molI],scalebar=scalebar[molI],\
                            distance=distance,name=title[molI],imagestretch=fluxStretch,\
                            colororgray=colorOverlay,colormap=cmap,plotlabel=miscLabel1[molI],
                            continuumContourParams=continuumContourParams,
                            contContourColor=contContourColor,cont_rms=cont_rms,cont_contour=cont_contour,
                            misc=miscLabel2[molI],textcolor=txtColor,extension=extensions,\
                            showContourLabels=showContourLabels,fontSize=fontsize,\
                            contourSize=contourSize,figSize=figSize[molI], titlepos=titlepos,miscpos=miscpos,labelpos=labelpos,beam_pad=beam_pad, labelspace = labelspace, customMarker=customMarker, customMarkerSize=customMarkerSize, cfg = config)
                        if colorOverlay == 'true':
                            ite = 'color'
                            if plot.auto:
                                plot.textcolor = 'white'
                        else:
                            ite = 'bw'
                            if plot.auto:
                                plot.textcolor = 'black'
                        plot.drawContinuum()
                        plot.setupContinuum(True, continuumOverlay[0])
                        # plotting rb channels for all qualifiers specified
                        start = True
                        for qualI2,qualV2 in enumerate(qualV1):
                            for dopplerI,dopplerV in enumerate(('r','b')):
                                inputFname = fileStructure.replace('-M-',molV).replace('-D-',dopplerV)\
                                                          .replace('-G-',qualV2)
                                '''
                                print(colors[dopplerI][qualI2],start,\
                                    show[molI][dopplerI],inputFname,allowNegContours[molI],\
                                    *contours[molI][qualI1][qualI2][dopplerI],labels[qualI1][qualI2])
                                '''
                                plot.setupContours(colors[dopplerI][qualI2],qualI2 == 0,\
                                    show[molI][dopplerI],inputFname,allowNegContours[molI],\
                                    *contours[molI][qualI1][qualI2][dopplerI],labels[qualI1][qualI2])
                                start = False
                        # plotting markers and outflows 
                        if markers[0]:
                            for sourceI,sourceV in enumerate(markers[2]):
                                # plot marker
                                print(sourceV)
                                mRa,mDec = map(float,parseRaDec(sourceV[1]).split(' '))
                                plot.showSources(mRa,mDec,sType=sourceV[3],circle=sourceV[4],\
                                                 size=sourceV[5],color=sourceV[0],label=sourceV[2])
                                if markers[1]:
                                    plot.addLabel(sourceV[0],sourceV[2])
                                # plot outflow
                                if outflows[sourceI][0]:
                                    print(sourceV)
                                    plot.drawOutflow(mRa,mDec,outflows[sourceI][1],outflows[sourceI][2])

                        # now plotting final steps and save
                        if showLabels:
                            plot.misc = f'{labels[qualI1][qualI2]}'
                        if colorBars:
                            plot.showColorbar()

                        outputFname = fileStructure.split('/')[-1].replace('-M-',molV).replace('-D-','')\
                                                          .replace('-G-','')\
                                                          .replace('.fits',output[qualI1]+'_'+ite)
                        plot.plotFormat()
                        #plot.fig.subplots_adjust(bottom=0.15, left=0.15)
                        plot.save(outputFname, dpi=300)
                        plot = None

                else:
                    # setup plot
                    colorOverlay = continuumOverlay[1]
                    plot = mappingContours(show_beam,continuumOverlay[0],ra,dec,minpixval=continuumParams[0],\
                        maxpixval=continuumParams[1], size=imSize[molI],scalebar=scalebar[molI],\
                        distance=distance,name=title[molI],imagestretch=fluxStretch,\
                        colororgray=colorOverlay,colormap=cmap,plotlabel=miscLabel1[molI],\
                        contContourColor=contContourColor,cont_rms=cont_rms,cont_contour=cont_contour,
                        continuumContourParams=continuumContourParams,
                        misc=miscLabel2[molI],textcolor=txtColor,extension=extensions,\
                        showContourLabels=showContourLabels,fontSize=fontSize,\
                        contourSize=contourSize,figSize=figSize[molI], titlepos=titlepos,miscpos=miscpos,labelpos=labelpos,beam_pad=beam_pad, labelspace = labelspace, customMarker=customMarker, customMarkerSize=customMarkerSize, cfg = config)
                    if plot.auto:
                        plot.textcolor = 'black'
                    plot.drawContinuum()
                    plot.setupContinuum(True, continuumOverlay[0])
                    # plotting rb channels for all qualifiers specified
                    start = True
                    for qualI2,qualV2 in enumerate(qualV1):
                        for dopplerI,dopplerV in enumerate(('r','b')):
                            inputFname = fileStructure.replace('-M-',molV).replace('-D-',dopplerV)\
                                                      .replace('-G-',qualV2)
                            '''
                            print((colors[dopplerI][qualI2],start,\
                                    show[molI][dopplerI],inputFname,allowNegContours[molI],\
                                    *contours[molI][qualI1][qualI2][dopplerI],labels[qualI1][qualI2]))
                            '''
                            plot.setupContours(colors[dopplerI][qualI2], start,\
                                show[molI][dopplerI],inputFname,allowNegContours[molI],\
                                *contours[molI][qualI1][qualI2][dopplerI],labels[qualI1][qualI2])
                            start = False
                    # plotting markers and outflows
                    if markers[0]:
                        for sourceI,sourceV in enumerate(markers[2]):
                            # plot marker
                            mRa,mDec = map(float,parseRaDec(sourceV[1]).split(' '))
                            plot.showSources(mRa,mDec,sType=sourceV[3],circle=sourceV[4],\
                                             size=sourceV[5],color=sourceV[0],label=sourceV[2])
                            if markers[1]:
                                plot.addLabel(sourceV[0],sourceV[2])
                            # plot outflow
                            if outflows[sourceI][0]:
                                print(sourceV)
                                plot.drawOutflow(mRa,mDec,outflows[sourceI][1],outflows[sourceI][2])

                    # now plotting final steps and save
                        if showLabels:
                            plot.misc = f'{labels[qualI1][qualI2]}'
                    if colorBars:
                        plot.showColorbar()

                    outputFname = "./"+fileStructure.split('/')[-1].replace('-M-',molV).replace('-D-','')\
                                                      .replace('-G-','')\
                                                      .replace('.fits',output[qualI1])
                    plot.plotFormat()
                    #plot.fig.subplots_adjust(bottom=0.15, left=0.15)
                    plot.save(outputFname,dpi=300)
                    plot = None
    else:
        #for molI,molV in enumerate(mols):
        for molI in range(mols):
            ra,dec = map(float,parseRaDec(center[molI]).split(' '))
            if continuumOverlay[1] == 'both':
                for colorOverlay in ('true','false'):
                    # setup plot
                    plot = mappingContours(show_beam,continuumOverlay[0],ra,dec,minpixval=continuumParams[0],\
                        maxpixval=continuumParams[1], size=imSize[molI],scalebar=scalebar[molI],\
                        distance=distance,name=title[molI],imagestretch=fluxStretch,\
                        colororgray=colorOverlay,colormap=cmap,plotlabel=miscLabel1[molI],\
                        contContourColor=contContourColor,cont_rms=cont_rms,cont_contour=cont_contour,
                        continuumContourParams=continuumContourParams,
                        misc=miscLabel2[molI],textcolor=txtColor,extension=extensions,
                        showContourLabels=showContourLabels,fontSize=fontSize,\
                        contourSize=contourSize,figSize=figSize[molI], titlepos=titlepos,miscpos=miscpos,labelpos=labelpos,beam_pad=beam_pad, labelspace = labelspace, customMarker=customMarker, customMarkerSize=customMarkerSize, cfg = config)
                    if colorOverlay == 'true':
                        ite = 'color'
                        if plot.auto:
                            plot.textcolor = 'white'
                    else:
                        ite = 'bw'
                        if plot.auto:
                            plot.textcolor = 'black'
                    plot.drawContinuum()
                    plot.setupContinuum(True,continuumOverlay[0])
                    # plotting markers and outflows 
                    if markers[0]:
                        for sourceI,sourceV in enumerate(markers[2]):
                            # plot marker
                            mRa,mDec = map(float,parseRaDec(sourceV[1]).split(' '))
                            plot.showSources(mRa,mDec,sType=sourceV[3],circle=sourceV[4],\
                                             size=sourceV[5],color=sourceV[0],label=sourceV[2])
                            if markers[1]:
                                plot.addLabel(sourceV[0],sourceV[2])
                            # plot outflow
                            if outflows[sourceI][0]:
                                print(sourceV)
                                plot.drawOutflow(mRa,mDec,outflows[sourceI][1],outflows[sourceI][2])

                    # now plotting final steps and save
                        if showLabels:
                            plot.misc = f'{labels[qualI1][qualI2]}'
                    if colorBars:
                        plot.showColorbar()

                    outputFname = "./"+continuumOverlay[0].split('/')[-1]\
                                                  .replace('.fits',output[molI]+'_'+ite)
                    plot.plotFormat()
                    #plot.fig.subplots_adjust(bottom=0.15, left=0.15)
                    plot.save(outputFname,dpi=300)
                    plot = None

            else:
                # setup plot
                colorOverlay = continuumOverlay[1]
                plot = mappingContours(show_beam,continuumOverlay[0],ra,dec,minpixval=continuumParams[0],\
                    maxpixval=continuumParams[1], size=imSize[molI],scalebar=scalebar[molI],\
                    distance=distance,name=title[molI],imagestretch=fluxStretch,\
                    colororgray=colorOverlay,colormap=cmap,plotlabel=miscLabel1[molI],\
                    continuumContourParams=continuumContourParams,
                    contContourColor=contContourColor,cont_rms=cont_rms,
                    cont_contour=cont_contour,
                    misc=miscLabel2[molI],textcolor=txtColor,extension=extensions,
                    showContourLabels=showContourLabels,fontSize=fontSize,\
                    contourSize=contourSize,figSize=figSize[molI], titlepos=titlepos,miscpos=miscpos,labelpos=labelpos,beam_pad=beam_pad, labelspace = labelspace, customMarker=customMarker, customMarkerSize=customMarkerSize, cfg = config)
                if plot.auto:
                    plot.textcolor = 'black'
                plot.drawContinuum()
                plot.setupContinuum(True,continuumOverlay[0])
                # plotting markers and outflows
                if markers[0]:
                    for sourceI,sourceV in enumerate(markers[2]):
                        # plot marker
                        print(f'On marker: {sourceV}')
                        mRa,mDec = map(float,parseRaDec(sourceV[1]).split(' '))
                        plot.showSources(mRa,mDec,sType=sourceV[3],circle=sourceV[4],\
                                         size=sourceV[5],color=sourceV[0],label=sourceV[2])
                        if markers[1]:
                            plot.addLabel(sourceV[0],sourceV[2])
                        # plot outflow
                        if outflows[sourceI][0]:
                            print(sourceV)
                            plot.drawOutflow(mRa,mDec,outflows[sourceI][1],outflows[sourceI][2])

                # now plotting final steps and save
                if colorBars:
                    plot.showColorbar()

                outputFname = "./"+continuumOverlay[0].split('/')[-1]\
                                                  .replace('.fits',output[molI])
                plot.plotFormat()
                #plot.fig.subplots_adjust(bottom=0.15, left=0.15)
                plot.save(outputFname,dpi=300)
                plot = None

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plots contours based on parameters found in the config file')
    parser.add_argument('--input', type=str,help='name of the config file')
    args = parser.parse_args()
    if args.input:
        main(args.input)
        print('Finished')
    else:
        print('No input')


# end of code

# end of file
