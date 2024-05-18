# Imports
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.stats
from astropy.io import fits 
import astropy.stats as stat
from astropy.stats import mad_std
from astropy.stats import sigma_clip
from photutils.utils import calc_total_error
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from photutils.detection import DAOStarFinder
import scipy.ndimage as interp

def fullReduce(frame,masterBiasPath,masterDarkPath,masterFlatPath,writeFinal=True):
    '''
    Reduces a frame of bias, dark, and flat.
    
    Parameters
    ----------
    frame:
        A string of the directory for the frame to be reduced.
        
    masterBiasPath:
        A string of the directory for the Master Bias file.
        
    masterDarkPath:
        A string of the directory for the Master Dark file.
        
    masterFlatPath:
        A string of the directory for the Master Flat file.
        
    writeFinal (optional):
        A Boolean determining whether the final reduced frame should be written as a file in the directory of the original file. True by default.
        
    Returns:
        A 2D array of the fully reduced frame.
    '''
    bFrame=biasSubtract(frame,masterBiasPath)
    bFramePath=os.path.dirname(frame)+'/b_'+os.path.basename(frame)
    dbFrame=darkSubtract(bFramePath,masterDarkPath)
    dbFramePath=os.path.dirname(frame)+'/db_'+os.path.basename(frame)
    if writeFinal==True:
        fdbFrame=flatDivide(dbFramePath,masterFlatPath,True)
    else:
        fdbframe=flatDivide(dbFramePath,masterFlatPath,False)
    return fdbFrame

def masterBias(biasFiles):
    '''
    Writes a Master Bias file given bias inputs.
    
    Parameters
    ----------
    biasFiles:
        A list-like object containing the directories of bias frames as strings.
        
    Returns:
        Zero.
    '''
    fits.writeto(os.path.dirname(biasFiles[0])+'/MasterBias.fit',medianStack(biasFiles),fits.getheader(biasFiles[0]),overwrite=True)
    return 0

def masterDark(darkFiles,masterBiasPath,exposure):
    '''
    Writes a Master Dark file given dark inputs and a Master Bias. Requires exposure to name the file.
    
    Parameters
    ----------
    darkFiles:
        A list-like object containing the directories of dark frames as strings.
        
    masterBiasPath:
        A string containing the directory of the Master Bias.
        
    exposure:
        The exposure of the dark files.
        
    Returns:
        Zero.
    '''
    for i in range (len(darkFiles)):
        biasSubtract(darkFiles[i],masterBiasPath)
    newDarkFiles=glob.glob(os.path.dirname(darkFiles[0])+'/b_*.fit')
    fits.writeto(os.path.dirname(darkFiles[0])+'/MasterDark'+str(exposure)+'s.fit',medianStack(newDarkFiles),fits.getheader(newDarkFiles[0]),overwrite=True)
    return 0

def customMasterDark(masterDarkPath,originalExposure,scaleFactor):
    '''
    Writes a custom Master Dark file that is a scaled variant of an existing Master Dark.
    
    Parameters
    ----------
    masterDarkPath:
        A string containing the directory of the Master Dark that the custom Master Dark will stem from.
        
    originalExposure:
        The exposure of the Master Dark.
        
    scaleFactor:
        The value in which the original Master Dark will be scaled.
        
    Returns:
        Zero.
    '''
    masterDarkData=fits.getdata(masterDarkPath)
    masterDarkHeader=fits.getheader(masterDarkPath)
    ySize,xSize=masterDarkData.shape
    newDark=np.zeros((ySize,xSize))
    newDark[:,:]=masterDarkData[:,:]*scaleFactor
    newExposure=originalExposure*scaleFactor
    fits.writeto(os.path.dirname(masterDarkPath)+'/MasterDark'+str(newExposure)+'s.fit',newDark,masterDarkHeader,overwrite=True)
    return 0

def masterFlat(flatFiles,masterBiasPath,masterDarkPath,filt):
    '''
    Writes a Master Flat file given a Master Bias and a Master Dark. Requires filter to name the file.
    
    Parameters
    ----------
    flatFiles:
        A list-like object containing the directories of flat frames as strings.
        
    masterBiasPath:
        A string containing the directory of the Master Bias.
        
    masterDarkPath:
        A string containing the directory of the Master Dark.
        
    filt:
        A string containing the filter for the flat.
        
    Returns:
        Zero.
    '''
    for i in range (len(flatFiles)):
        biasSubtract(flatFiles[i],masterBiasPath)
    newFlatFiles=glob.glob(os.path.dirname(flatFiles[0])+'/b_*.fit')
    for i in range (len(newFlatFiles)):
        darkSubtract(newFlatFiles[i],masterDarkPath)
    newFlatFiles=glob.glob(os.path.dirname(flatFiles[0])+'/db_*.fit')
    fits.writeto(os.path.dirname(flatFiles[0])+'/Master'+str(filt)+'Flat.fit',normalizedMedianStack(newFlatFiles),fits.getheader(newFlatFiles[0]),overwrite=True)
    return 0

def medianStack(files):
    '''
    Takes a list of frames and combines the median of each measurement into a single frame.
    
    Parameters
    ----------
    files:
        A list-like object containing the directories of the files as strings.
        
    Returns:
        A 2D-array in which all values are medians of that pixel in the input files.
    '''
    fileCount=len(files)
    firstFrame=fits.getdata(files[0])
    ySize,xSize=firstFrame.shape
    fitsHolder=np.zeros((ySize,xSize,fileCount))
    for i in range(0,fileCount):
        image=fits.getdata(files[i])
        fitsHolder[:,:,i]=image
    medianFrame=np.nanmedian(fitsHolder,axis=2)
    return medianFrame

def normalizedMedianStack(files):
    '''
    Takes a list of frames and combines the median of each measurement into a single normalized frame.
    
    Parameters
    ----------
    files:
        A list-like object containing the directories of the files as strings.
        
    Returns:
        A normalized 2D-array in which all values are medians of that pixel in the input files.
    '''
    frameCount=len(files)
    firstFrame=fits.getdata(files[0])
    ySize,xSize=firstFrame.shape
    fitsHolder=np.zeros((ySize,xSize,frameCount))
    for i in range(0,frameCount):
        image=fits.getdata(files[i])
        normalizedImage=image/np.median(image)
        fitsHolder[:,:,i]=normalizedImage
    normalizedMedianFrame=np.median(fitsHolder,axis=2)
    return normalizedMedianFrame

def biasSubtract(file,biasPath,write=True):
    '''
    Takes some frame and a bias frame to create a FITS file of the original frame with bias subtracted.
    
    Parameters
    ----------
    file:
        A string containing the directory of the file.
        
    biasPath:
        A string containing the directory of the bias frame.
        
    write (optional):
        A Boolean determining whether a new file should be written in the directory of the file. True by default.
        
    Returns:
        A 2D-array in which all values are the input pixel value minus the bias of that pixel.
    '''
    rawFrame=fits.getdata(file)
    biasFrame=fits.getdata(biasPath)
    ySize,xSize=rawFrame.shape
    biasSubtractedFrame=np.zeros((ySize,xSize))
    biasSubtractedFrame[:,:]=rawFrame[:,:]-biasFrame[:,:]
    if write==True:
        frameHeader=fits.getheader(file)
        fileDestination=os.path.dirname(file)+'/b_'+os.path.basename(file)
        fits.writeto(fileDestination,biasSubtractedFrame,frameHeader,overwrite=True)
    return biasSubtractedFrame

def darkSubtract(file,darkPath,write=True):
    '''
    Takes some frame and a dark frame to create a FITS file of the original frame with dark subtracted.
    
    Parameters
    ----------
    file:
        A string containing the directory of the file.
        
    darkPath:
        A string containing the directory of the dark frame.
        
    write (optional):
        A Boolean determining whether a new file should be written in the directory of the file. True by default.
        
    Returns:
        A 2D-array in which all values are the input pixel value minus the dark of that pixel.
    '''
    rawFrame=fits.getdata(file)
    darkFrame=fits.getdata(darkPath)
    ySize,xSize=rawFrame.shape
    darkSubtractedFrame=np.zeros((ySize,xSize))
    darkSubtractedFrame[:,:]=rawFrame[:,:]-darkFrame[:,:]
    if write==True:
        frameHeader=fits.getheader(file)
        fileDestination=os.path.dirname(file)+'/d'+os.path.basename(file)
        fits.writeto(fileDestination,darkSubtractedFrame,frameHeader,overwrite=True)
    return darkSubtractedFrame

def flatDivide(file,flatPath,write=True):
    '''
    Takes some frame and a flat frame to create a FITS file of the original frame with flat data reduced.
    
    Parameters
    ----------
    file:
        A string containing the directory of the file.
        
    flatPath:
        A string containing the directory of the flat frame.
        
    write (optional):
        A Boolean determining whether a new file should be written in the directory of the file. True by default.
        
    Returns:
        A 2D-array in which all values are the input pixel value divided by the flat of that pixel.
    '''
    rawFrame=fits.getdata(file)
    flatFrame=fits.getdata(flatPath)
    ySize,xSize=rawFrame.shape
    flatDividedFrame=np.zeros((ySize,xSize))
    flatFrame[flatFrame[:,:]==0]=1
    flatDividedFrame[:,:]=rawFrame[:,:]/flatFrame[:,:]
    if write==True:
        frameHeader=fits.getheader(file)
        fileDestination=os.path.dirname(file)+'/f'+os.path.basename(file)
        fits.writeto(fileDestination,flatDividedFrame,frameHeader,overwrite=True)
    return flatDividedFrame

def centroid(image,bgSample):
    '''
    Takes an image and a background sample and returns the approximate centroid location.
    
    Parameters
    ----------
    image:
        A 2D array containing the image information.
        
    bgSample:
        A 2D array containing the background information.
        
    Returns:
        A 2-component tuple (yCenter,xCenter) containing the y-center value and the x-center value.
    '''
    bgMean=astropy.stats.sigma_clipped_stats(bgSample)[0]
    bgStanDev=astropy.stats.sigma_clipped_stats(bgSample)[2]
    xCenTop=0
    yCenTop=0
    bottom=0
    for j in range(0,image.shape[0]):
        for i in range(0,image.shape[1]):
            if image[j,i]>=(3*bgStanDev):
                xCenTop+=(i+1)*(image[j,i]-bgMean)
                yCenTop+=(j+1)*(image[j,i]-bgMean)
                bottom+=image[j,i]-bgMean
    xCen=xCenTop/bottom
    yCen=yCenTop/bottom
    return yCen,xCen

def cropData(frame,sideLength,centerCoords,explicit=True):
    '''
    Returns a cropped image specified by the length of the sides and the coordinates of the center. If the side length is even, it will be reduced to the next lowest odd integer.
    
    Parameters
    ----------
    frame:
        A 2D array of the frame.
        
    sideLength:
        An integer specifying the side length of the crop window.
        
    centerCoords:
        A 2-component tuple (y,x) specifying the center position of the crop window relative to the original frame.
        
    explicit (optional):
        A Boolean stating whether the input frame is a 2D array. If False, will assume it is a directory. True by default.
        
    Returns:
        A 2D-array that contains all the information within the specified crop window.
    '''
    if explicit==False:
        image=fits.getdata(frame)
    else:
        image=np.copy(frame)
    croppedImage=np.zeros((sideLength,sideLength))
    if sideLength%2==0:
        sideLength-=1
    for j in range(sideLength):
        for i in range(sideLength):
            croppedImage[j,i]=image[int((centerCoords[0]-((sideLength-1)/2))+j),int((centerCoords[1]-((sideLength-1)/2))+i)]
    return croppedImage

def trimData(image,trim,explicit=True):
    '''
    Returns a trimmed image in which the edges are shaved by the specified pixel trim amount.
    
    Parameters
    ----------
    image:
        A 2D array containing the image information.
        
    trim:
        An integer specifying the amount to trim off of the edges of the image.
        
    explicit (optional):
        A Boolean stating whether the input frame is a 2D array. If False, will assume it is a directory. True by default.
        
    Returns:
        A 2D-array that contains all the information within the specified trim window.
    '''
    if explicit==False:
        image=fits.getdata(image)
    trimmedImage=np.zeros((image.shape[0]-(2*trim),image.shape[1]-(2*trim)))
    for j in range(image.shape[0]-(2*trim)):
        for i in range(image.shape[1]-(2*trim)):
            trimmedImage[j,i]=image[j+trim,i+trim]
    return trimmedImage

def findShift(image1,image2,bg,sideLength,centerCoords,explicit=True):
    '''
    Returns the pixel shift (y,x) between two images based on centroid calculations.
    
    Parameters
    ----------
    image1:
        A 2D array containing information for the first image.
        
    image2:
        A 2D array containing information for the second image.
        
    bg:
        A 2D array containing a sample of background.
        
    sideLength:
        An integer specifying the side length of the window in which the shift will be found.
        
    centerCoords:
        A 2-component tuple (y,x) specifying the center coordinate of the window in which the shift will be found.
        
    explicit (optional):
        A Boolean stating whether the input frame is a 2D array. If False, will assume it is a directory. True by default.
        
    Returns:
        A 2-component tuple (y,x) that is the shift between the two images.
    '''
    if explicit==False:
        image1=np.copy(fits.getdata(image1))
        image2=np.copy(fits.getdata(image1))
    croppedImage1=cropData(image1,sideLength,centerCoords)
    croppedImage2=cropData(image2,sideLength,centerCoords)
    image1Center=centroid(croppedImage1,bg)
    image2Center=centroid(croppedImage2,bg)
    x=image2Center[1]-image1Center[1]
    y=image2Center[0]-image1Center[0]
    return y,x

def imageStack(imageList,bg,windowWidth,refCenterCoords,orderFix=False):
    '''
    Takes a list of frames, a background sample, and a reference window's width and center coordinates to return an image that stacks all frames based on centroiding of the first file in the list.
    
    Parameters
    ----------
    imageList:
        A list-like object containing the directories of all files as strings.
        
    bg :
        A 2D array containing a sample of background.
        
    windowWidth:
        An integer specifying the side length of the window in which the shift will be found.
        
    refCenterCoords:
        A 2-component tuple (y,x) specifying the center coordinate of the window in which the shift will be found.
        
    orderFix:
        A Boolean determining whether to use order=0 for files that return all NaNs in shift. False by default.
        
    Returns:
        A 2-component tuple (y,x) that is the shift between the two images.
    '''
    header=fits.getheader(imageList[0])
    base=fits.getdata(imageList[0])
    shiftList=[]
    shiftListX=[]
    shiftListY=[]
    for i in range(1,len(imageList)):
        im=fits.getdata(imageList[i])
        shift=findShift(base,im,bg,windowWidth,refCenterCoords)
        shiftList.append(shift)
        shiftListX.append(abs(shift[1]))
        shiftListY.append(abs(shift[0]))
    xPad=np.max(shiftListX)
    yPad=np.max(shiftListY)
    padding=int(np.max([xPad,yPad]))+1
    paddedBase=np.pad(base,padding,'constant',constant_values=-1000)
    stackArray=np.zeros((base.shape[0]+(2*padding),base.shape[1]+(2*padding),len(imageList)-1))
    for i in range(1,len(imageList)):
        rawFrame=fits.getdata(imageList[i])
        paddedFrame=np.pad(rawFrame,padding,'constant',constant_values=-1000)
        if orderFix==True:
            shiftedFrame=interp.shift(paddedFrame,(-1*shiftList[i-1][0],-1*shiftList[i-1][1]),cval=-1000,order=0)
        else:
            shiftedFrame=interp.shift(paddedFrame,(-1*shiftList[i-1][0],-1*shiftList[i-1][1]),cval=-1000)
        shiftedFrame[shiftedFrame<=-500]=np.nan
        stackArray[:,:,i-1]=shiftedFrame[:,:]
    finalImage=np.nanmedian(stackArray,axis=2)
    header=fits.getheader(imageList[0])
    fits.writeto(os.path.dirname(imageList[0])+'/stacked_'+os.path.basename(imageList[0]),finalImage,header,overwrite=True)
    print ('Successful stack!')
    return

def imageAlign(imageList,bg,windowWidth,refCenterCoords,orderFix=False):
    '''
    Takes a list of frames, a background sample, and a reference window's width and center coordinates to align frames based on centroiding of the first file in the list.
    
    Parameters
    ----------
    imageList:
        A list-like object containing the directories of all files as strings.
        
    bg :
        A 2D array containing a sample of background.
        
    windowWidth:
        An integer specifying the side length of the window in which the shift will be found.
        
    refCenterCoords:
        A 2-component tuple (y,x) specifying the center coordinate of the window in which the shift will be found.
        
    orderFix:
        A Boolean determining whether to use order=0 for files that return all NaNs in shift. False by default.
        
    Returns:
        None.
    '''
    header=fits.getheader(imageList[0])
    base=fits.getdata(imageList[0])
    shiftList=[]
    shiftListX=[]
    shiftListY=[]
    for i in range(1,len(imageList)):
        im=fits.getdata(imageList[i])
        shift=findShift(base,im,bg,windowWidth,refCenterCoords)
        shiftList.append(shift)
        shiftListX.append(abs(shift[1]))
        shiftListY.append(abs(shift[0]))
    xPad=np.max(shiftListX)
    yPad=np.max(shiftListY)
    padding=int(np.max([xPad,yPad]))+1
    paddedBase=np.pad(base,padding,'constant',constant_values=np.min(base)-1000)
    for i in range(1,len(imageList)):
        rawFrame=fits.getdata(imageList[i])
        paddedFrame=np.pad(rawFrame,padding,'constant',constant_values=-1000)
        if orderFix==True:
            shiftedFrame=interp.shift(paddedFrame,(-1*shiftList[i-1][0],-1*shiftList[i-1][1]),cval=-1000,order=0)
        else:
            shiftedFrame=interp.shift(paddedFrame,(-1*shiftList[i-1][0],-1*shiftList[i-1][1]),cval=-1000)
        shiftedFrame[shiftedFrame<=-500]=np.nan
        header=fits.getheader(imageList[i])
        fits.writeto(os.path.dirname(imageList[i])+'/a'+os.path.basename(imageList[i]),shiftedFrame,header,overwrite=True)
    fits.writeto(os.path.dirname(imageList[0])+'/a'+os.path.basename(imageList[0]),base,header,overwrite=True)
    print ('Successful align!')
    return

def manualImageAlign(image1,image2,shift,padding=-1,orderFix=False):
    '''
    Takes two frames and manual a shift value to align frames.
    
    Parameters
    ----------
    image1:
        A string directory containing the alignment image.
        
    image2:
        A string directory containing the image that will be aligned to the first image.
        
    shift:
        A 2-component tuple (y,x)
        
    padding:
        An integer that determines the appropriate padding on the shifted images. By default, it is the highest axial shift rounded up to the nearest integer.
        
    orderFix:
        A Boolean determining whether to use order=0 for files that return all NaNs in shift. False by default.
        
    Returns:
        None
    '''
    if padding==-1:
        padding=int(np.max(shift))+1
    base=fits.getdata(image1)
    paddedBase=np.pad(base,padding,'constant',constant_values=np.min(base)-1000)
    rawFrame=fits.getdata(image2)
    paddedFrame=np.pad(rawFrame,padding,'constant',constant_values=-1000)
    if orderFix==True:
        shiftedFrame=interp.shift(paddedFrame,shift,cval=-1000,order=0)
    else:
        shiftedFrame=interp.shift(paddedFrame,shift,cval=-1000)
    shiftedFrame[shiftedFrame<=-500]=np.nan
    header=fits.getheader(image2)
    fits.writeto(os.path.dirname(image2)+'/a'+os.path.basename(image2),shiftedFrame,header,overwrite=True)
    print ('Successful align!')
    return

def bgErrorEstimate(fitsfile):
    """
    This function estimates the background in order to find the uncertainty in each pixel and returns an array representing this
    It writes two new files: 1 with the uncertainty of the background, and one with the uncertainty of the entire input image
    
    Parameters
    ----------
    fitsfile - string
        File path of a fits file
    
    Returns
    -------
    error_image - array
        The array representing the uncertainty in each pixel of the input
    """
    fitsdata=fits.getdata(fitsfile)
    hdr=fits.getheader(fitsfile)
    
    filtered_data=sigma_clip(fitsdata,sigma=3.,copy=False)
    
    bkg_values_nan=filtered_data.filled(fill_value=np.nan)
    bkg_error=np.sqrt(bkg_values_nan)
    bkg_error[np.isnan(bkg_error)]=np.nanmedian(bkg_error)
    
    print("Writing the background-only error image: ",fitsfile.split('.')[0]+"_bgerror.fit")
    fits.writeto(fitsfile.split('.')[0]+"_bgerror.fit",bkg_error,hdr, overwrite=True)
    
    effective_gain=1.4 # electrons per ADU
    
    error_image=calc_total_error(fitsdata,bkg_error,effective_gain)  
    
    print("Writing the total error image: ",fitsfile.split('.')[0]+"_error.fit")
    fits.writeto(fitsfile.split('.')[0]+"_error.fit", error_image, hdr, overwrite=True)
    
    return error_image

def measurePhotometry(fitsfile, star_xpos, star_ypos, aperture_radius, sky_inner, sky_outer, error_array):
    """
    Makes a table with the apertures of the stars
    
    Parameters
    ----------
    fitsfile - string
        File path of a fits file
    star_xpos - array
        The array of the x positions of stars
    star_ypos - array
        The array of the y positions of stars
    aperture_radius - float
        The radius of the aperture
        
    Returns
    -------
    xpos - array
        The x positions of the stars found in an array
    ypos - array
        The y positions of the stars found in an array  
    """
    # Reads in data from fits file
    image = fits.getdata(fitsfile)
    star_pos = np.vstack([star_xpos, star_ypos]).T
    
    starapertures = CircularAperture(star_pos,r = aperture_radius)
    skyannuli = CircularAnnulus(star_pos, r_in = sky_inner, r_out = sky_outer)
    phot_apers = [starapertures, skyannuli]
    
    # What is new about the way we're calling aperture_photometry?
    # Makes a table of the apertures of the image
    phot_table = aperture_photometry(image, phot_apers, error=error_array)
        
    # Calculate mean background in annulus and subtract from aperture flux
    bkg_mean = phot_table['aperture_sum_1'] / skyannuli.area
    bkg_starap_sum = bkg_mean * starapertures.area
    final_sum = phot_table['aperture_sum_0']-bkg_starap_sum
    phot_table['Star Counts (No BG)'] = final_sum
    
    # Finds the error in aperture
    bkg_mean_err = phot_table['aperture_sum_err_1'] / skyannuli.area
    bkg_sum_err = bkg_mean_err * starapertures.area
    
    # Finds the uncertenty of the aperture
    phot_table['Star Counts Error (No BG)'] = np.sqrt((phot_table['aperture_sum_err_0']**2)+(bkg_sum_err**2)) 
    
    return phot_table

# Star extraction function -- this function can be to also return the x and y positions to the notebook to use later:
def starExtractor(fitsfile, nsigma_value, fwhm_value):
    """
    Makes a file with the positions of the centroid of each star.
    
    Parameters
    ----------
    fitsfile - string
        File path of a fits file
    nsigma_value - float
        The amount of sigmas that will be found for the threshold
    fwhm_vale - float
        The value of the FWHM
        
    Returns
    -------
    xpos - array
        The x positions of the stars found in an array
    ypos - array
        The y positions of the stars found in an array  
    """
    
    # First, check if the region file exists yet, so it doesn't get overwritten
    regionfile=fitsfile.split(".")[0]+".reg"
    
    # Read in the data from the fits file ***
    image=fits.getdata(fitsfile)
    
    # Measure the median absolute standard deviation of the image:
    bkg_sigma=stat.median_absolute_deviation(image,ignore_nan=True)

    # *** Define the parameters for DAOStarFinder
    # fwhm: Full-With, Half-Max: the length at half the max
    # threshold: How bright the objects need to be to be considered stars
    daofind=DAOStarFinder(fwhm=fwhm_value,threshold=nsigma_value*bkg_sigma,exclude_border=True)
    
    # Apply DAOStarFinder to the image
    sources=daofind(image)
    nstars=len(sources)
    print("Number of stars found in ",fitsfile,":",nstars)
    
    # Define arrays of x-position and y-position
    xpos=np.array(sources['xcentroid'])
    ypos = np.array(sources['ycentroid'])
    
    # Write the positions to a .reg file based on the input file name
    if os.path.exists(regionfile):
        os.remove(regionfile)
    
    f=open(regionfile,'w') 
    for i in range(0,len(xpos)):
        f.write('circle '+str(xpos[i])+' '+str(ypos[i])+' '+str(fwhm_value)+'\n')
    f.close()
    print("Wrote ",regionfile)
    return xpos,ypos # Return the x and y positions of each star as variables

def calcInstMags(data,exptime):
    '''

    Takes a dataset containing B, V, and R fluxes and their errors errors along with an exposure time to calculate instrumental magnitudes.

    '''

    data['BfluxPerSec']=data['Bflux']/exptime
    data['BfluxPerSec_Err'] = abs(data['Bflux_Err']/exptime)
    data['Binst']=-2.5*np.log10(abs(data['Bflux']))
    data['Binst_Err']=abs(2.5*0.434*data['Bflux_Err']/data['Bflux'])
    
    data['VfluxPerSec']=data['Vflux']/exptime
    data['VfluxPerSec_Err'] = abs(data['Vflux_Err']/exptime)
    data['Vinst']=-2.5*np.log10(abs(data['Vflux']))
    data['Vinst_Err']=abs(2.5*0.434*data['Vflux_Err']/data['Vflux'])
         
    data['RfluxPerSec']=data['Rflux']/exptime
    data['RfluxPerSec_Err'] = abs(data['Rflux_Err']/exptime)
    data['Rinst']=-2.5*np.log10(abs(data['Rflux']))
    data['Rinst_Err']=abs(2.5*0.434*data['Rflux_Err']/data['Rflux'])

    return data