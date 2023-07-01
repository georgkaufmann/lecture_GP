"""
GP
library for field work
2023-06-30
Georg Kaufmann
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from matplotlib.ticker import LogFormatter 

#================================#
def readTopography(fileTopo,path='./',iskip=10,control=False,plot=False):
    """
    Read topography data from geodyn5 file

    Parameters
    ----------
    fileTopo : str
        Full name of input file (topography in geodyn5 format)
    path : str 
        Path to input file, default: ./
    iskip : int
        Read in only iskip line, default: 10
    control : bool
        Control output, default: None
    plot : bool
        plot map, default: None

    Returns
    -------
    easting : 1D numpy array
        List of easting coordinates [m]
    northing : 1D numpy array
        List of northing coordinates [m]
    elevation : 1D numpy array
        List of elevations [m]

    Notes
    -----

    Raises
    ------
    ValueError
        if path/fileTopo does not exist, aborts
    """
    import numpy as np
    import os.path, sys
    # check, if path+fileTopo exists
    if (os.path.isfile(path+fileTopo)==False):
        print ('File ',path+fileTopo,' does not exist, aborted')
        sys.exit()
    else:
        # read all lines from geodyn5 elevation file
        f = open(path+fileTopo,'r')
        topolines = f.readlines()
        f.close()
        # create data fields (datetime, easting, northing, elevation)
        imeta=0; itopodata=0; i=0
        datetime  =[]
        easting   = np.empty(0)
        northing  = np.empty(0)
        elevation = np.empty(0)
        # go through lines, separate meta-data from data, fill fields, skip every iskip lines
        for line in topolines:
            # Get next line from file
            if (line[0] == '!'):
                imeta += 1
                #print(line.split())
            else:
                i += 1
                if (i%iskip == 0):
                    itopodata += 1
                    easting = np.append(easting,[float(line.split()[1])],axis=0)
                    northing = np.append(northing,[float(line.split()[2])],axis=0)
                    elevation = np.append(elevation,[float(line.split()[3])],axis=0)
                    datetime.append(line.split()[0])
        # control output
        if (control):
            print('File read:                 ',path+fileTopo)
            print('Number of topo lines:      ',len(topolines))
            print('Number of meta-data read:  ',imeta)
            print('Number of topo data read:  ',itopodata)
            print('min/max easting:           ',easting.min(),easting.max())
            print('min/max northing:          ',northing.min(),northing.max())
            print('min/max elevation:         ',elevation.min(),elevation.max())
        # plot topography
        if (plot):
            plt.figure(figsize=(10,10))
            ax = plt.axes(aspect='equal')
            plt.title('Topography: '+fileTopo)
            # draw ticks and labels
            ax.set_xlim([easting.min(),easting.max()])
            ax.set_ylim([northing.min(),northing.max()])
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Northing [m]')
            ax.ticklabel_format(useOffset=False)
            # topography as image
            vmin=elevation.min();vmax=elevation.max()
            bounds = np.linspace(vmin,vmax,20)
            cbar1=ax.tricontourf(easting,northing,elevation,cmap='terrain',vmin=vmin,vmax=vmax,levels=bounds)
            plt.colorbar(cbar1,label='Elevation [m]',shrink=0.6)
            # topo contours
            cbar2=ax.tricontour(easting,northing,elevation,levels=bounds,colors='black',linewidths=1,alpha=0.3)
            ax.clabel(cbar2,colors='black',inline=True,fmt='{:.0f}'.format)
        return easting,northing,elevation

#================================#
def readBouguerAnomaly(fileGrav,path='./',iskip=1,control=False,plot=False):
    """
    Read Bouguer anomaly data from geodyn5 file

    Parameters
    ----------
    fileGrav : str
        Full name of input file (in geodyn5 format)
    path : str 
        Path to input file, default: ./
    iskip : int
        Read in only iskip line, default: 10
    control : bool
        Control output, default: None
    plot : bool
        plot map, default: None

    Returns
    -------
    easting : 1D numpy array
        List of easting coordinates [m]
    northing : 1D numpy array
        List of northing coordinates [m]
    topo : 1D numpy array
        List of elevations [m]
    boug : 1D numpy array
        List of elevations [m]

    Notes
    -----

    Raises
    ------
    ValueError
        if path/fileGrav does not exist, aborts
    """
    import numpy as np
    import os.path, sys
    # check, if path+fileTopo exists
    if (os.path.isfile(path+fileGrav)==False):
        print ('File ',path+fileGrav,' does not exist, aborted')
        sys.exit()
    else:
        # read all lines from geodyn5 elevation file
        f = open(path+fileGrav,'r')
        topolines = f.readlines()
        f.close()
        # create data fields (datetime, easting, northing, elevation)
        imeta=0; itopodata=0; i=0
        datetime  =[]
        easting   = np.empty(0)
        northing  = np.empty(0)
        topo      = np.empty(0)
        boug      = np.empty(0)
        # go through lines, separate meta-data from data, fill fields, skip every iskip lines
        for line in topolines:
            # Get next line from file
            if (line[0] == '!'):
                imeta += 1
                #print(line.split())
            else:
                i += 1
                if (i%iskip == 0):
                    itopodata += 1
                    easting  = np.append(easting,[float(line.split()[1])],axis=0)
                    northing = np.append(northing,[float(line.split()[2])],axis=0)
                    topo     = np.append(topo,[float(line.split()[3])],axis=0)
                    boug     = np.append(boug,[float(line.split()[5])],axis=0)
                    datetime.append(line.split()[0])
        # control output
        if (control):
            print('File read:                 ',path+fileGrav)
            print('Number of topo lines:      ',len(topolines))
            print('Number of meta-data read:  ',imeta)
            print('Number of topo data read:  ',itopodata)
            print('min/max easting:           ',easting.min(),easting.max())
            print('min/max northing:          ',northing.min(),northing.max())
            print('min/max elevation:         ',topo.min(),topo.max())
            print('min/max Bouguer anomaly:   ',boug.min(),boug.max())
        # plot Bouguer anomaly
        if (plot):
            plt.figure(figsize=(10,10))
            ax = plt.axes(aspect='equal')
            plt.title('Bouguer anomaly: '+fileGrav)
            # draw ticks and labels
            ax.set_xlim([easting.min(),easting.max()])
            ax.set_ylim([northing.min(),northing.max()])
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Northing [m]')
            ax.ticklabel_format(useOffset=False)
            # topography as image
            vmin=boug.min();vmax=boug.max()
            bounds = np.linspace(vmin,vmax,20)
            cbar1=ax.tricontourf(easting,northing,boug,cmap='seismic',vmin=vmin,vmax=vmax,levels=bounds)
            plt.colorbar(cbar1,label='Bouguer anomaly [mGal]',shrink=0.6)
            # topo contours
            cbar2=ax.tricontour(easting,northing,boug,levels=bounds,colors='black',linewidths=1,alpha=0.3)
            ax.clabel(cbar2,colors='black',inline=True,fmt='{:.0f}'.format)
        return easting,northing,topo,boug

#================================#
def readTotalField(fileMAG,path='./',iskip=1,control=False,plot=False):
    """
    Read Total-field anomaly data from geodyn5 file

    Parameters
    ----------
    fileMAG : str
        Full name of input file (in geodyn5 format)
    path : str 
        Path to input file, default: ./
    iskip : int
        Read in only iskip line, default: 10
    control : bool
        Control output, default: None
    plot : bool
        plot map, default: None

    Returns
    -------
    easting : 1D numpy array
        List of easting coordinates [m]
    northing : 1D numpy array
        List of northing coordinates [m]
    topo : 1D numpy array
        List of elevations [m]
    total : 1D numpy array
        List of total-field values [m]

    Notes
    -----

    Raises
    ------
    ValueError
        if path/fileMAG does not exist, aborts
    """
    import numpy as np
    import os.path, sys
    # check, if path+fileMAG exists
    if (os.path.isfile(path+fileMAG)==False):
        print ('File ',path+fileMAG,' does not exist, aborted')
        sys.exit()
    else:
        # read all lines from geodyn5 elevation file
        f = open(path+fileMAG,'r')
        topolines = f.readlines()
        f.close()
        # create data fields (datetime, easting, northing, topo, total)
        imeta=0; itopodata=0; i=0
        datetime  =[]
        easting   = np.empty(0)
        northing  = np.empty(0)
        topo      = np.empty(0)
        total     = np.empty(0)
        # go through lines, separate meta-data from data, fill fields, skip every iskip lines
        for line in topolines:
            # Get next line from file
            if (line[0] == '!'):
                imeta += 1
                #print(line.split())
            else:
                i += 1
                if (i%iskip == 0):
                    itopodata += 1
                    easting  = np.append(easting,[float(line.split()[1])],axis=0)
                    northing = np.append(northing,[float(line.split()[2])],axis=0)
                    topo     = np.append(topo,[float(line.split()[3])],axis=0)
                    total     = np.append(total,[float(line.split()[5])],axis=0)
                    datetime.append(line.split()[0])
        # control output
        if (control):
            print('File read:                 ',path+fileMAG)
            print('Number of topo lines:      ',len(topolines))
            print('Number of meta-data read:  ',imeta)
            print('Number of topo data read:  ',itopodata)
            print('min/max easting:           ',easting.min(),easting.max())
            print('min/max northing:          ',northing.min(),northing.max())
            print('min/max elevation:         ',topo.min(),topo.max())
            print('min/max Total field:       ',total.min(),total.max())
        # plot Bouguer anomaly
        if (plot):
            plt.figure(figsize=(10,10))
            ax = plt.axes(aspect='equal')
            plt.title('Total-field anomaly: '+fileMAG)
            # draw ticks and labels
            ax.set_xlim([easting.min(),easting.max()])
            ax.set_ylim([northing.min(),northing.max()])
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Northing [m]')
            ax.ticklabel_format(useOffset=False)
            # topography as image
            vmin=total.min();vmax=total.max()
            bounds = np.linspace(vmin,vmax,20)
            cbar1=ax.tricontourf(easting,northing,total,cmap='seismic',vmin=vmin,vmax=vmax,levels=bounds)
            plt.colorbar(cbar1,label='Total-field anomaly [nT]',shrink=0.6)
            # topo contours
            cbar2=ax.tricontour(easting,northing,total,levels=bounds,colors='black',linewidths=1,alpha=0.3)
            ax.clabel(cbar2,colors='black',inline=True,fmt='{:.0f}'.format)
        return easting,northing,topo,total

#================================#
def readERTprofile(fileERT,path='./',control=False,plot=False):
    """
    Read ERT data data from geodyn5 file

    Parameters
    ----------
    fileERT : str
        Full name of input file (in geodyn5 format)
    path : str 
        Path to input file, default: ./
    control : bool
        Control output, default: None
    plot : bool
        plot map, default: None

    Returns
    -------
    ert : 2D numpy array
        List of easting, northing, elevation, offset [m], rho [Ohmm], profile [m]
    points : 2D numpy array
        List of profile, offset [m]
    tri : object
        trianulation object
    total : 1D numpy array
        List of total-field values [m]

    Notes
    -----

    Raises
    ------
    ValueError
        if path/fileMAG does not exist, aborts
    """
    import numpy as np
    import os.path, sys
    # check, if path+fileMAG exists
    if (os.path.isfile(path+fileERT)==False):
        print ('File ',path+fileERT,' does not exist, aborted')
        sys.exit()
    else:
        # read all lines from geodyn5 file
        f = open(path+fileERT, 'r')
        ertlines = f.readlines()
        f.close()
        # create ert data fields (easting, northing, elev/boug)
        imeta=0; iertdata=0
        datetime =[]
        ert = np.empty((0,6))
        # go through lines, separate meta-data from data, fill fields
        for line in ertlines:
            # Get next line from file
            if (line[0] == '!'):
                imeta += 1
                #print(line.split())
            else:
                iertdata += 1
                add  = np.array([[float(line.split()[1]),
                                float(line.split()[2]),
                                float(line.split()[3]),
                                float(line.split()[4]),
                                float(line.split()[5]),
                                float(line.split()[6])]])
                ert = np.append(ert,add,axis=0)
                datetime.append(line.split()[0])
        # control output
        if (control):
            print('Number of meta-data lines: ',imeta)
            print('Number of ert lines:       ',iertdata)
            print('min/max elevation:         ',ert[:,2].min(),ert[:,2].max())
            print('min/max offset:            ',ert[:,3].min(),ert[:,3].max())
            print('min/max resistivity:       ',ert[:,4].min(),ert[:,4].max())
            print('min/max profile:           ',ert[:,5].min(),ert[:,5].max())
            print(ert.shape)
        # triangulation points
        points = np.zeros(ert.shape[0]*2).reshape(ert.shape[0],2)
        for i in range(ert.shape[0]):
            points[i][0] = ert[i,5]
            points[i][1] = ert[i,3]
        if (control):
            print(points.shape)
        # Delaunay triangulation 
        tri = scipy.spatial.Delaunay(points)
        # plot profile
        if (plot):
            fig,ax1 = plt.subplots(1,1,figsize=(12.0, 4.0))

            ax1.set_xlabel('Profile distance [m]')
            ax1.set_ylabel('Elevation [m]')
            ax1.set_title('ERT profile: '+fileERT)
            ax1.set_xlim([ert[:,5].min(),ert[:,5].max()])
            ax1.set_ylim([ert[:,2].min()+ert[:,3].min(),ert[:,2].max()])
        
            sorted = np.argsort(ert[:,5])
            ax1.plot(ert[sorted,5],ert[sorted,2],linewidth=4,color='black')
            contour1=plt.tricontour(points[:,0], points[:,1]+ert[:,2], tri.simplices,ert[:,4],
                                    levels=[10,20,50,100,200,500,1000,2000,3000,5000],
                                    colors='black',alpha=0.5,linewidths=1)
            ax1.clabel(contour1,colors='black',inline=True,fmt='{:.0f}'.format)
            im1 = plt.tripcolor(points[:,0], points[:,1]+ert[:,2], tri.simplices,ert[:,4], shading='gouraud',
                                cmap = plt.get_cmap('rainbow'),norm=matplotlib.colors.LogNorm(vmin=ert[:,4].min(),vmax=ert[:,4].max()))
            formatter = LogFormatter(10, labelOnlyBase=False,minor_thresholds=(3,2)) 
            cbar1=fig.colorbar(im1,ax=ax1,shrink=0.9,ticks=[10,20,50,100,200,500,1000,5000],format=formatter)
            cbar1.set_label('Specific resistivity [\u03A9'+'m]', rotation=90)
        return ert,points,tri

#================================#
def createERTCoordElevation(nameERT,nElectrodes,sElectrodes,GPSPoints,easting,northing,elevation,path='./',control=False,plot=False):
    """
    Read GPS coordinates taken along ERT profile, 
    create coordinates for every electrode position
    and interpolate elevation for electrode from topo data
    
    Parameters
    ----------
    nameERT : str
        name of ERT profile
    nElectrodes : int
        number of electrodes along profile
    sElectrodes : int
        electrode spacing [m]
    GPSPoints : 2D float array 
        List of GPS points easting,northing) along profile
    easting : 1D float array
        List of easting coordinates [m] (from readTopography.py)
    northing : 1D float array
        List of northing coordinates [m] (from readTopography.py)
    elevation : 1D float array
        List of elevations [m] (from readTopography.py)
    control : bool
        Control output, default: None
    plot : bool
        plot simple map, default: None

    Returns
    -------
    elecPoints : 2D float array
        List of coordinates, elevations, and distances for profile electrodes [m]
        elecPoints[:,0] - easting [m]
        elecPoints[:,1] - northing [m]
        elecPoints[:,2] - elevation [m]
        elecPoints[:,3] - distance [m]

    Notes
    -----
    Sample parameter values for 25 electrodes, 3m spacing, and two GPS points:
        nElectrodes = 25
        sElectrodes = 3
        GPSPoints = np.array([
        [605503.44,5714171.93],
        [605450.70,5714227.35]
        ])
    """
    import numpy as np
    import scipy.interpolate
    import os.path, sys
    print('nameERT:                   ',nameERT)
    print('nElectrodes:               ',nElectrodes)
    print('sElectrodes:               ',sElectrodes)
    # calculate linear distance between GPS points
    GPSDistance = np.zeros(len(GPSPoints[0]))
    sProfile    = (nElectrodes-1)*sElectrodes
    print('GPS points:                ',len(GPSPoints[0]))
    print('Profile length:            ',sProfile,' m')
    for i in range(1,len(GPSDistance)):
        GPSDistance[i] = np.sqrt((GPSPoints[i-1][0]-GPSPoints[i][0])**2 + (GPSPoints[i-1][1]-GPSPoints[i][1])**2)
    # calculate electrode distance and positions 
    elecPoints   = np.zeros([nElectrodes,4])
    for i in range(nElectrodes):
        elecPoints[i,3] = GPSDistance[0] + i*sElectrodes
        elecPoints[i,0] = np.interp(elecPoints[i,3], GPSDistance,GPSPoints[:,0])
        elecPoints[i,1] = np.interp(elecPoints[i,3], GPSDistance,GPSPoints[:,1])
    # calculate electrode elevations from interpolation
    elecPoints[:,2] = scipy.interpolate.griddata(np.c_[easting,northing], elevation, (elecPoints[:,0], elecPoints[:,1]), method='linear')
    if (control):
        for i in range(nElectrodes):
            print("%4i %12.8f %12.8f %8.2f %8.2f" % (i,elecPoints[i,0],elecPoints[i,1],elecPoints[i,2],elecPoints[i,3]))
    # save coordinates, distance, elevation to profile file
    f = open(path+nameERT+'.profile','w')
    for i in range(nElectrodes):
        print("%12.8f %12.8f %8.2f %8.2f" % (elecPoints[i,0],elecPoints[i,1],elecPoints[i,3],elecPoints[i,2]),file=f)
    f.close()
    print('nameERT:                   ',nameERT)
    print('File written:              ',path+nameERT+'.profile')
    # plot profile
    if (plot):
        fig,axs = plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
        # plot map view for profile
        axs[0].set_title('ERT profile: '+nameERT)
        axs[0].set_aspect('equal')
        axs[0].set_xlim([elecPoints[:,0].min()-10,elecPoints[:,0].max()+10])
        axs[0].set_ylim([elecPoints[:,1].min()-10,elecPoints[:,1].max()+10])
        axs[0].set_xlabel('Easting [m]')
        axs[0].set_ylabel('Northing [m]')
        axs[0].plot(elecPoints[:,0],elecPoints[:,1],lw=2,marker='o',color='blue')
        axs[0].plot(GPSPoints[:,0],GPSPoints[:,1],lw=0,marker='x',color='red')
        # plot elevation of profile
        axs[1].set_xlim([0,sProfile])
        axs[1].set_ylim([elecPoints[:,2].min(),elecPoints[:,2].max()])
        axs[1].set_xlabel('Profile [m]')
        axs[1].set_ylabel('Elevation [m]')
        axs[1].plot(elecPoints[:,3],elecPoints[:,2],lw=2,marker='o',color='blue')
    return elecPoints

#================================#
def createGPRCoordElevation(nameGPR,lProfile,sProfile,GPSPoints,easting,northing,elevation,traceInc=0,path='./',control=False,plot=False):
    """
    Read GPS coordinates taken along GPR profile, 
    create coordinates for every nth trace icrement
    and interpolate elevation for traces from topo data
    
    Parameters
    ----------
    nameGPR : str
        name of GPR profile
    lProfile : float
        length of GPR profile  [m]
    sProfile : float
        spacing distance for profile [m]
    GPSPoints : 2D float array 
        List of GPS points easting,northing) along profile
    easting : 1D float array
        List of easting coordinates [m] (from readTopography.py)
    northing : 1D float array
        List of northing coordinates [m] (from readTopography.py)
    elevation : 1D float array
        List of elevations [m] (from readTopography.py)
    traceInc : float
        trace increment (from GPS recording)
    control : bool
        Control output, default: None
    plot : bool
        plot simple map, default: None

    Returns
    -------
    gprPoints : 2D float array
        List of coordinates, elevations, and distances for profile electrodes [m]
        gprPoints[:,0] - easting [m]
        gprPoints[:,1] - northing [m]
        gprPoints[:,2] - elevation [m]
        gprPoints[:,3] - distance [m]

    Notes
    -----
    Sample parameter values for 100m long GPR profile, sample every 1m, trace increment 0.01, and two GPS points:
        lProfile = 100.
        sProfile = 1.
        traceInc = 0.01
        GPSPoints = np.array([
        [605503.44,5714171.93],
        [605450.70,5714227.35]
        ])
    """
    import numpy as np
    import scipy.interpolate
    import os.path, sys
    if (traceInc==0):
        print('Trace increment not set')
        sys.exit()
    nTraces = int(lProfile/sProfile)
    print('nameGPR:                   ',nameGPR)
    print('lProfile:                  ',lProfile)
    print('sProfile:                  ',sProfile)
    print('nTraces:                   ',nTraces)
    print('traceInc:                  ',traceInc)
    # calculate linear distance between GPS points
    GPSDistance = np.zeros(len(GPSPoints[0]))
    print('GPS points:                ',len(GPSPoints[0]))
    for i in range(1,len(GPSDistance)):
        GPSDistance[i] = np.sqrt((GPSPoints[i-1][0]-GPSPoints[i][0])**2 + (GPSPoints[i-1][1]-GPSPoints[i][1])**2)
    # calculate electrode distance and positions 
    gprPoints   = np.zeros([nTraces,4])
    for i in range(nTraces):
        gprPoints[i,3] = GPSDistance[0] + i*sProfile
        gprPoints[i,0] = np.interp(gprPoints[i,3], GPSDistance,GPSPoints[:,0])
        gprPoints[i,1] = np.interp(gprPoints[i,3], GPSDistance,GPSPoints[:,1])
    # calculate electrode elevations from interpolation
    gprPoints[:,2] = scipy.interpolate.griddata(np.c_[easting,northing], elevation, (gprPoints[:,0], gprPoints[:,1]), method='linear')
    if (control):
        for i in range(nTraces):
            print("%4i %12.8f %12.8f %8.2f %8.2f" % (i,gprPoints[i,0],gprPoints[i,1],gprPoints[i,2],gprPoints[i,3]))
    # save coordinates, distance, elevation to profile file
    f = open(path+nameGPR+'.utm','w')
    for i in range(nTraces):
        print("%10i %12.8f %12.8f %8.2f" % (1+int(gprPoints[i,3]/traceInc),gprPoints[i,1],gprPoints[i,0],gprPoints[i,2]),file=f)
    f.close()
    print('nameGPR:                   ',nameGPR)
    print('File written:              ',path+nameGPR+'.utm')
    # plot profile
    if (plot):
        fig,axs = plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
        # plot map view for profile
        axs[0].set_title('GPR profile: '+nameGPR)
        axs[0].set_aspect('equal')
        axs[0].set_xlim([gprPoints[:,0].min()-10,gprPoints[:,0].max()+10])
        axs[0].set_ylim([gprPoints[:,1].min()-10,gprPoints[:,1].max()+10])
        axs[0].set_xlabel('Easting [m]')
        axs[0].set_ylabel('Northing [m]')
        axs[0].plot(gprPoints[:,0],gprPoints[:,1],lw=2,marker='o',markersize=3,color='blue')
        axs[0].plot(GPSPoints[:,0],GPSPoints[:,1],lw=0,marker='x',color='red')
        # plot elevation of profile
        axs[1].set_xlim([0,lProfile])
        axs[1].set_ylim([gprPoints[:,2].min(),gprPoints[:,2].max()])
        axs[1].set_xlabel('Profile [m]')
        axs[1].set_ylabel('Elevation [m]')
        axs[1].plot(gprPoints[:,3],gprPoints[:,2],lw=2,marker='o',markersize=3,color='blue')
    return gprPoints

#================================#
def createGravityCoordElevation(fileGrav,path='./',irepeat=1,control=False,plot=False):
    """
    Read coordinates of gravity stations along with leveling data
    
    Parameters
    ----------
    fileGrav : str
        Full name of input file (gravity coord file)
    path : str 
        Path to input file, default: ./
    irepeat : int
        Repeat cooordinate irepeat times, default: 1
    control : bool
        Control output, default: None
    plot : bool
        plot map, default: None

    Returns
    -------
    easting : 1D numpy array
        List of easting coordinates [m]
    northing : 1D numpy array
        List of northing coordinates [m]
    elevation : 1D numpy array
        List of elevations [m]

    Notes
    -----
    sample input file. 
    INFO lines not needed
    BASE line needed, holds reference elevation
    DATA line(s) needed, holds point coordinates, for target point ('to') and back- and forward view from leveling
    
        ~~~~~~~~~~~~~~~
        INFO Base point with absolute coordinates
        BASE    G01     none    51.155896000564098 10.03715998493135 300
        INFO Points levelled
        DATA    G01     Z01     51.155896000564098 10.03715998493135   13.2 4.5
        DATA    Z01     G02     51.155896000564098 10.03715998493135   30.6 24.4
        ~~~~~~~~~~~~~~~
    
    Raises
    ------
    ValueError
        if path+fileGrav does not exist, aborts
    """
    import numpy as np
    import os.path, sys
    # check, if path+fileGrav exists
    if (os.path.isfile(path+fileGrav)==False):
        print ('File ',path+fileGrav,' does not exist, aborted')
        sys.exit()
    else:
        # read file content into object
        file_grav = open(path+fileGrav,'r')
        # convert object into a list containing all lines
        lines = file_grav.readlines()
        file_grav.close()
        print('File read:                 ',path+fileGrav)
        lon   = np.empty(0)
        lat   = np.empty(0)
        topo  = np.empty(0)
        # go through the list line by line
        ibase = 0; idata=0
        coords = dict(dict());
        add   = dict();
        for line in lines:
            if (line[0:4] == 'BASE'):
                ibase += 1
                splitted = line.split()
                add   = dict();
                add['from'] = splitted[1]
                add['to'] = splitted[2]
                add['lat'] = splitted[3]
                add['lon'] = splitted[4]
                add['elev'] = splitted[5]
                coords[splitted[1]] = add
            elif (line[0:4] == 'DATA'):
                idata += 1
                splitted = line.split()
                add   = dict();
                add['from'] = splitted[1]
                add['to'] = splitted[2]
                add['lat'] = splitted[3]
                add['lon'] = splitted[4]
                add['elev'] = -99999
                add['backward'] = splitted[5]
                add['forward'] = splitted[6]
                coords[splitted[2]] = add
        print('BASE lines:                ',ibase)
        print('Data lines:                ',idata)
        # calculate elevation difference form leveling data and save coordinates to file
        if (ibase == 0):
            print ('No base point defined, aborted')
        else:
            outfile = path+fileGrav.split('.')[0]+'.xyz'
            f = open(outfile, 'w')
            icoord = 0
            for key in coords:
                if (icoord == 0):
                    if (control):
                        print("%10s %14.10f %14.10f %8.2f %5s" % (coords[key]['from'],float(coords[key]['lon']),float(coords[key]['lat']),float(coords[key]['elev']),'fixed'))
                else:
                    if coords[key]['from'] in coords:
                        diff = float(coords[key]['backward']) - float(coords[key]['forward'])
                        elev = float(coords[coords[key]['from']]['elev']) + 0.1*diff
                        coords[key]['elev'] = elev
                        lon  = np.append(lon,[float(coords[key]['lon'])],axis=0)
                        lat  = np.append(lat,[float(coords[key]['lat'])],axis=0)
                        topo = np.append(topo,[float(coords[key]['elev'])],axis=0)
                        if (control):
                            print("%10s %14.10f %14.10f %8.2f" % (coords[key]['to'],float(coords[key]['lon']),float(coords[key]['lat']),float(coords[key]['elev'])))
                        for i in range(irepeat):
                            print("%14.10f %14.10f %8.2f" % (float(coords[key]['lon']),float(coords[key]['lat']),float(coords[key]['elev'])),file=f)
                    else:
                        print('Wrong network connection!'+coords[key]['from']+coords[key]['to']);break
                icoord += 1
        print('File written:              ',outfile)
        # plot topography
        if (plot):
            plt.figure(figsize=(10,10))
            ax = plt.axes(aspect='equal')
            plt.title('Leveling: '+fileGrav)
            # draw ticks and labels
            ax.set_xlim([lon.min(),lon.max()])
            ax.set_ylim([lat.min(),lat.max()])
            ax.set_xlabel('Longitude [$^{\circ}$]')
            ax.set_ylabel('Latitude [$^{\circ}$]')
            ax.ticklabel_format(useOffset=False)
            # topography as image
            vmin=topo.min();vmax=topo.max()
            bounds = np.linspace(vmin,vmax,20)
            cbar1=ax.tricontourf(lon,lat,topo,cmap='terrain',vmin=vmin,vmax=vmax,levels=bounds)
            plt.colorbar(cbar1,label='Elevation [m]',shrink=0.6)
            # topo contours
            cbar2=ax.tricontour(lon,lat,topo,levels=bounds,colors='black',linewidths=1,alpha=0.3)
            ax.clabel(cbar2,colors='black',inline=True,fmt='{:.0f}'.format)
            # gravity stations
            ax.plot(lon,lat,lw=0,marker='o',markersize=4,color='red',alpha=0.4)
        return

#================================#
def addERTCoordElevation(fileERT,elecPoints,path='./',control=False):
    """
    Add profile length and elevation along ERT profile
    to Res2DInv file
    
    Parameters
    ----------
    fileERT : str
        file name of ERT profile in Res2DInv format
    elecPoints : 2D float array
        List of coordinates, elevations, and distances for profile electrodes [m]
        elecPoints[:,0] - easting [m]
        elecPoints[:,1] - northing [m]
        elecPoints[:,2] - elevation [m]
        elecPoints[:,3] - distance [m]
    path : str 
        Path to input file, default: ./
    control : bool
        Control output, default: None
        
    Returns
    -------
    -none-

    Notes
    -----
    Read the ERT file in Res2DInv format and replaces last section with elevation data
    """
    import numpy as np
    import os.path, sys
    # check, if path+fileERT exists
    if (os.path.isfile(path+fileERT)==False):
        print ('File ',path+fileERT,' does not exist, aborted')
        sys.exit()
    else:
        # read file content into lines object
        f = open(path+fileERT, 'r')
        # convert object into a list containing all lines
        lines = f.readlines()
        f.close() 
        print('File read:                 ',path+fileERT)
        # get 6 header lines, extract electrode spacing, type of ERT profile, number of measurements
        sElectrodes   = float(lines[1])
        typeERT       = int(lines[2])
        nMeasurements = int(lines[3])
        if (control):
            print('Electode spacing:          ',sElectrodes)
            print('type of ERT measurments:   ',typeERT)
            print('Number of measurements:    ',nMeasurements)
        # save measurements to array
        ERTData = np.zeros([nMeasurements,4])
        for i in range(nMeasurements):
            ERTData[i,0] = lines[6+i].replace(',','').split()[0]
            ERTData[i,1] = lines[6+i].replace(',','').split()[1]
            ERTData[i,2] = lines[6+i].replace(',','').split()[2]
            ERTData[i,3] = lines[6+i].replace(',','').split()[3]
        if (control):
            print(ERTData)
        # check if topography is already present
        itopo = int(lines[nMeasurements+6])
        if (itopo != 0):
            print('Topography data already present in '+path+fileERT)
            #sys.exit()
        else:
            # replace last section with elevation data
            f = open(path+fileERT.split('.')[0]+'_mod.dat', 'w')
            for i in range(nMeasurements+6):
                print(lines[i].replace(',',''),end="",file=f)
            print(2,file=f)
            print(elecPoints.shape[0],file=f)
            for i in range(elecPoints.shape[0]):
                print(elecPoints[i,3],elecPoints[i,2],file=f)
            print(0,file=f)
            print('File written:              ',path+fileERT.split('.')[0]+'_mod.dat')
        return

#================================#
