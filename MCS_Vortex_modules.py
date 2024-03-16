##Binning modules used for the MCS profile data:

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d as bin2d
from scipy import interpolate
import scipy.ndimage as ndimage
from operator import itemgetter
from planetThermo import pco2
from planetThermo import tco2
from tqdm import tqdm


#Altitude and Longitude binning (Zonal mean with altitude)
def Alt_Long_binning(dfc,choice, longbinsize, longmin, longmax):
    '''
    bins the data by LS (per 1 LS) and then bins the temp, dust and H2Oice quantities
    in a 2D meshgrid of altitude (per km) and longitude(per longbinsize degrees).
    
    inputs: 
    - MCS dataframe that includes: TEMPERATURE, PRESSURE, DUST ,ALTITUDE, LATITUDE, LONGITUDE, H2OICE, LS, LOCALTIME
    - choice of which variable to bin from 'temp', 'dust', 'h2oice', 'pressure' 
    - longbinsize = bin size in longitude
    - longmin = minimum longitude
    - longmax = maximum longitude
    (note that altitude bins are always 1km in this by default)

    
    output: 
    - 1D altitude bins array
    - 1D longitude bins array
    - a dictionary structure with each entry representing 1 LS containing a matrix of
    temperature (or dust or h2oice) quantities in each meshpoint 
    to call one LS entry in the dictionary : dict['LS'] (this is a matrix with size len(altbinsarray)xlen(longbinsarray))
    '''

    #creating the base bins (won't change for the different LS's)
    
    #setting altitude bins
    #using overall max and min of all data read 
    #step size is set to 1km 
    A_bins = np.linspace(0,int(np.ceil(dfc.ALTITUDE.max())),int(np.ceil(dfc.ALTITUDE.max()))+1)
    
    #setting longitude bins
    #bin size determine by input
    Lmin = longmin
    Lmax = longmax

    L_bins = np.linspace(Lmin, Lmax,int(((Lmax - Lmin)/longbinsize))+1)

    #Setting Ls bins
    #determining LS span (minimum and maximum)
    LSmin = int(round(dfc.LS.min()))
    LSmax = int(round(dfc.LS.max()))
    #How many Ls bins total: 
    #this will be used to create array of LSs
    LSsiz  = (LSmax - LSmin)
    #creating array of Ls with spacing = binsize
    LS_arr = np.linspace(LSmin, LSmax, LSsiz+1)
    LS_arr[LS_arr!= LSmax] #removing extra entry from entry
    
    #empty dictionary for storing purposes
    results = dict() 
    
    #looping through the different LS bins to and performing the 2d binning with pressure and lat for each LS
    for i in range(int(LSsiz)):
        #pick out only entries with that specific Ls
        dfc1 = dfc[(dfc.LS >= LS_arr[i]) & (dfc.LS < LS_arr[i] + 1)]
        #ignore if empty
        if  dfc1.shape[0] <= 10:
            continue
        
        #binning temperature
        if choice == 'temp':
            temp = bin2d(dfc1.LONGITUDE, dfc1.ALTITUDE, dfc1.TEMPERATURE, 
                         bins = [L_bins,A_bins], statistic = 'mean')[0]
            ans = temp.T
            
        #binning dust
        elif choice == 'dust':
            dust = bin2d(dfc1.LONGITUDE, dfc1.ALTITUDE, dfc1.DUST,
                         bins = [L_bins,A_bins], statistic = 'mean')[0]
            ans = dust.T
            
        #binning H2O ice
        elif choice == 'h2oice':
            h2oice = bin2d(dfc1.LONGITUDE, dfc1.ALTITUDE, dfc1.H2OICE,
                                        bins = [L_bins,A_bins], statistic = 'mean')[0]
            ans = h2oice.T

        #binning Pressure:
        elif choice == 'pressure':
            press = bin2d(dfc1.LONGITUDE, dfc1.ALTITUDE, dfc1.PRESSURE,
                                        bins = [L_bins,A_bins], statistic = 'mean')[0]
            ans = press.T
            
        #error message if no choice was made
        else: 
            print('invalid variable input please choose from the following: temp, dust, h2oice ')
           
        #save current Ls binned data to dictionary    
        results[str(LS_arr[i])] = ans
    return L_bins, A_bins, results

def North_Polar_Column_Density(df, latmin,latmax,latbinsiz,longmin,longmax,longbinsiz):
    ''' 
    Function to compute the column density of CO2 clouds in the polar vortex

    inputs: 
    - MCS dataframe that includes: TEMPERATURE, PRESSURE, DUST ,ALTITUDE, LATITUDE, LONGITUDE, H2OICE, LS, LOCALTIME 
    - latmin = minimum latitude bound
    - latmax = maximum latitude bound
    - latbinsiz = size of latitude bin step
    - longmin = minimum longitude bound
    - longmax = maximum longitude bound
    - longbinsiz = size of longitude bin step

    outputs:
    - dataframe containing the following columns: 
            ColDen = column density of CO2 in (m-2) at that mesh point
            Ls = Ls of mesh point
            Lat = latitude of mesh point
            Long = longitude of mesh point

    '''
    #define constants/inputs (using km as base length)
    Qc = 3.0249 #efficiency of extinction for CO2
    ac = 1e-5 #m particle cross section
    rhoc = 1600 #kg/m3 #solid density of CO2 ice (Find reference)
    R_m = 3.3895e6 #radius of Mars in m
    h = 1000 #m # vertical height of 1 pixel
    mu = 8.5e-6 #(kg/ m s) OR (Pa s) # molecular viscosity of atmosphere at 150 K
    G = 6.67408e-11 #m3/kg s  # gravitational constant
    Kb = 1.38064852e-23 #m2 kg/ s2 K #boltzmann constant
    dc = 3.3e-10 #m CO2 molecule diameter
    
    latrange = np.arange(latmin,latmax+1,latbinsiz)
    
    # create dataframe that limits by latitude: 
    PV = df[(df.LS <= 360)&(df.LS >=180)&(df.ALTITUDE <= 60)&(df.ALTITUDE >= 10)&(df.LATITUDE >= latrange[0])&(df.LATITUDE <= latrange[1])]

    #resetting index of new dataframe
    PV = PV.reset_index()
    PV = PV.drop(['index'],axis = 1)
    #calculate the frost point for each entry
    PV['T_frost'] = tco2(0.95*PV.PRESSURE)
    # create a new dataframe that only includes the CO2 measurements,, which is anywhere where the recorded 
    #temp is equal to or less than the frost point
    PVC = PV[(PV.TEMPERATURE <= PV.T_frost+5)]
    #resetting index of new dataframe
    PVC = PVC.reset_index()
    PVC = PVC.drop(['index'],axis = 1)
    #bin the data
    LongV, AltV, co2ice = Alt_Long_binning(PVC,'dust',longbinsiz,longmin,longmax) #here, dust is actually co2 since we've seperated out the co2 from the dust data
    names = list(co2ice.keys())
              
    #find the column density of CO2
    Numden = dict() #no density of CO2 particles
    
    for i in tqdm(range(len(co2ice))):
        # redefine the array of co2ice value
        C = co2ice[names[i]]*1e-3 #m^-1 (multiply by 1e-3 to convert from km-1 to m-1)

        #step 1: convert the co2 ice opacities to optical depth
        C = C*h
        #step 2: calculate the no. density of CO2 ice particles in each pixel

        numden = 2.3*C/(Qc*np.pi*(ac**2)*h) #scaling the opacity CO2/dust ratio = 2.3 (source: Dave's work for Paul and Carlos's paper)

        #remove all nan values
        numden[np.isnan(numden)] = 0  
        Numden[names[i]] = numden
              
    nonzeros = Numden[names[0]].copy()
    nonzeros[Numden[names[0]] == 0] = np.nan
    column_density = np.nansum(nonzeros,axis=0)*1000 #unit is m-2
    oldDF = pd.DataFrame(column_density)
    oldDF.columns = ['ColDen']
    oldDF['Ls'] = int(float(names[0]))
    oldDF['Lat'] = latrange[0]
    oldDF['Long'] = LongV[:-1]
    
    for i in range(1,len(names)):
        nonzeros = Numden[names[i]].copy()
        nonzeros[Numden[names[i]] == 0] = np.nan
        column_density = np.nansum(nonzeros,axis=0)*1000

        newDF = pd.DataFrame(column_density)
        newDF.columns = ['ColDen']
        newDF['Ls'] = int(float(names[i]))
        newDF['Lat'] = latrange[0]
        newDF['Long'] = LongV[:-1]

        tempDF = pd.concat([oldDF, newDF], ignore_index = True)
        tempDF.reset_index()

        oldDF = tempDF
        
    for i in tqdm(range(1,len(latrange)-1)):
        
        # create a new dataframe that only includes the polar vortex of the north pole region (North Polar Vortex)
        PV = df[(df.LS <= 360)&(df.LS >=180)&(df.ALTITUDE <= 60)&(df.ALTITUDE >= 10)&(df.LATITUDE >= latrange[i])&(df.LATITUDE <= latrange[i+1])]
        #resetting index of new dataframe
        PV = PV.reset_index()
        PV = PV.drop(['index'],axis = 1)
        #calculate the frost point for each entry
        PV['T_frost'] = tco2(0.95*PV.PRESSURE)
        # create a new dataframe that only includes the CO2 measurements,, which is anywhere where the recorded 
        #temp is equal to or less than the frost point
        PVC = PV[(PV.TEMPERATURE <= PV.T_frost+5)]
        #resetting index of new dataframe
        PVC = PVC.reset_index()
        PVC = PVC.drop(['index'],axis = 1)
        #bin the data
        LongV, AltV, co2ice = Alt_Long_binning(PVC,'dust',longbinsiz,longmin,longmax) #here, dust is actually co2 since we've seperated out the co2 from the dust data
        names = list(co2ice.keys())
        
        #find the column density of CO2
        Numden = dict() #no density of CO2 particles
        for j in tqdm(range(len(co2ice))):
            # redefine the array of co2ice value
            C = co2ice[names[j]]*1e-3 #m^-1 (multiply by 1e-3 to convert from km-1 to m-1)
            
            #step 1: convert the co2 ice opacities to optical depth
            C = C*h
            #step 2: calculate the no. density of CO2 ice particles in each pixel
            
            numden = 2.3*C/(Qc*np.pi*(ac**2)*h) #scaling the opacity CO2/dust ratio = 2.3 (source: Dave's work for Paul and Carlos's paper)
            
            #remove all nan values
            numden[np.isnan(numden)] = 0  
        
        
            Numden[names[j]] = numden
            
        for k in range(0,len(names)):
            nonzeros = Numden[names[k]].copy()
            nonzeros[Numden[names[k]] == 0] = np.nan
            column_density = np.nansum(nonzeros,axis=0)*1000
            
            newDF = pd.DataFrame(column_density)
            newDF.columns = ['ColDen']
            newDF['Ls'] = int(float(names[k]))
            newDF['Lat'] = latrange[i]
            newDF['Long'] = LongV[:-1]
            
            tempDF = pd.concat([oldDF, newDF], ignore_index = True)
            tempDF.reset_index()
            
            oldDF = tempDF
            
    Polar_Clouds = tempDF
    return Polar_Clouds 

def South_Polar_Column_Density(df, latmin,latmax,latbinsiz,longmin,longmax,longbinsiz):
    ''' 
    Function to compute the column density of CO2 clouds in the polar vortex

    inputs: 
    - MCS dataframe that includes: TEMPERATURE, PRESSURE, DUST ,ALTITUDE, LATITUDE, LONGITUDE, H2OICE, LS, LOCALTIME 
    - latmin = minimum latitude bound
    - latmax = maximum latitude bound
    - latbinsiz = size of latitude bin step
    - longmin = minimum longitude bound
    - longmax = maximum longitude bound
    - longbinsiz = size of longitude bin step

    outputs:
    - dataframe containing the following columns: 
            ColDen = column density of CO2 in (m-2) at that mesh point
            Ls = Ls of mesh point
            Lat = latitude of mesh point
            Long = longitude of mesh point

    '''
    #define constants/inputs (using km as base length)
    Qc = 3.0249 #efficiency of extinction for CO2
    ac = 1e-5 #m particle cross section
    rhoc = 1600 #kg/m3 #solid density of CO2 ice (Find reference)
    R_m = 3.3895e6 #radius of Mars in m
    h = 1000 #m # vertical height of 1 pixel
    mu = 8.5e-6 #(kg/ m s) OR (Pa s) # molecular viscosity of atmosphere at 150 K
    G = 6.67408e-11 #m3/kg s  # gravitational constant
    Kb = 1.38064852e-23 #m2 kg/ s2 K #boltzmann constant
    dc = 3.3e-10 #m CO2 molecule diameter
    
    latrange = np.arange(latmin,latmax+1,latbinsiz)
    
    # create dataframe that limits by latitude: 
    PV = df[(df.LS <= 180)&(df.LS >=0)&(df.ALTITUDE <= 60)&(df.ALTITUDE >= 10)&(df.LATITUDE >= latrange[0])&(df.LATITUDE <= latrange[1])]

    #resetting index of new dataframe
    PV = PV.reset_index()
    PV = PV.drop(['index'],axis = 1)
    #calculate the frost point for each entry
    PV['T_frost'] = tco2(0.95*PV.PRESSURE)
    # create a new dataframe that only includes the CO2 measurements,, which is anywhere where the recorded 
    #temp is equal to or less than the frost point
    PVC = PV[(PV.TEMPERATURE <= PV.T_frost+5)]
    #resetting index of new dataframe
    PVC = PVC.reset_index()
    PVC = PVC.drop(['index'],axis = 1)
    #bin the data
    LongV, AltV, co2ice = Alt_Long_binning(PVC,'dust',longbinsiz,longmin,longmax) #here, dust is actually co2 since we've seperated out the co2 from the dust data
    names = list(co2ice.keys())
              
    #find the column density of CO2
    Numden = dict() #no density of CO2 particles
    
    for i in tqdm(range(len(co2ice))):
        # redefine the array of co2ice value
        C = co2ice[names[i]]*1e-3 #m^-1 (multiply by 1e-3 to convert from km-1 to m-1)

        #step 1: convert the co2 ice opacities to optical depth
        C = C*h
        #step 2: calculate the no. density of CO2 ice particles in each pixel

        numden = 2.3*C/(Qc*np.pi*(ac**2)*h) #scaling the opacity CO2/dust ratio = 2.3 (source: Dave's work for Paul and Carlos's paper)

        #remove all nan values
        numden[np.isnan(numden)] = 0  
        Numden[names[i]] = numden
              
    nonzeros = Numden[names[0]].copy()
    nonzeros[Numden[names[0]] == 0] = np.nan
    column_density = np.nansum(nonzeros,axis=0)*1000 #unit is m-2
    oldDF = pd.DataFrame(column_density)
    oldDF.columns = ['ColDen']
    oldDF['Ls'] = int(float(names[0]))
    oldDF['Lat'] = latrange[0]
    oldDF['Long'] = LongV[:-1]
    
    for i in range(1,len(names)):
        nonzeros = Numden[names[i]].copy()
        nonzeros[Numden[names[i]] == 0] = np.nan
        column_density = np.nansum(nonzeros,axis=0)*1000

        newDF = pd.DataFrame(column_density)
        newDF.columns = ['ColDen']
        newDF['Ls'] = int(float(names[i]))
        newDF['Lat'] = latrange[0]
        newDF['Long'] = LongV[:-1]

        tempDF = pd.concat([oldDF, newDF], ignore_index = True)
        tempDF.reset_index()

        oldDF = tempDF
        
    for i in tqdm(range(1,len(latrange)-1)):
        
        # create a new dataframe that only includes the polar vortex of the north pole region (North Polar Vortex)
        PV = df[(df.LS <= 180)&(df.LS >=0)&(df.ALTITUDE <= 60)&(df.ALTITUDE >= 10)&(df.LATITUDE >= latrange[i])&(df.LATITUDE <= latrange[i+1])]
        #resetting index of new dataframe
        PV = PV.reset_index()
        PV = PV.drop(['index'],axis = 1)
        #calculate the frost point for each entry
        PV['T_frost'] = tco2(0.95*PV.PRESSURE)
        # create a new dataframe that only includes the CO2 measurements,, which is anywhere where the recorded 
        #temp is equal to or less than the frost point
        PVC = PV[(PV.TEMPERATURE <= PV.T_frost+5)]
        #resetting index of new dataframe
        PVC = PVC.reset_index()
        PVC = PVC.drop(['index'],axis = 1)
        #bin the data
        LongV, AltV, co2ice = Alt_Long_binning(PVC,'dust',longbinsiz,longmin,longmax) #here, dust is actually co2 since we've seperated out the co2 from the dust data
        names = list(co2ice.keys())
        
        #find the column density of CO2
        Numden = dict() #no density of CO2 particles
        for j in tqdm(range(len(co2ice))):
            # redefine the array of co2ice value
            C = co2ice[names[j]]*1e-3 #m^-1 (multiply by 1e-3 to convert from km-1 to m-1)
            
            #step 1: convert the co2 ice opacities to optical depth
            C = C*h
            #step 2: calculate the no. density of CO2 ice particles in each pixel
            
            numden = 2.3*C/(Qc*np.pi*(ac**2)*h) #scaling the opacity CO2/dust ratio = 2.3 (source: Dave's work for Paul and Carlos's paper)
            
            #remove all nan values
            numden[np.isnan(numden)] = 0  
        
        
            Numden[names[j]] = numden
            
        for k in range(0,len(names)):
            nonzeros = Numden[names[k]].copy()
            nonzeros[Numden[names[k]] == 0] = np.nan
            column_density = np.nansum(nonzeros,axis=0)*1000
            
            newDF = pd.DataFrame(column_density)
            newDF.columns = ['ColDen']
            newDF['Ls'] = int(float(names[k]))
            newDF['Lat'] = latrange[i]
            newDF['Long'] = LongV[:-1]
            
            tempDF = pd.concat([oldDF, newDF], ignore_index = True)
            tempDF.reset_index()
            
            oldDF = tempDF
            
    Polar_Clouds = tempDF
    return Polar_Clouds 

#Latitude and Longitude binning
def Lat_Long_binning(dfc,choice, longbinsize, latbinsize, Latmin,Latmax, Longmin,Longmax):
    '''
    bins the data by LS (per 1 LS) and then bins the temp, dust and H2Oice quantities
    in a 2D meshgrid of longitude (per longbinsiz deg) and latitude(per latbinsiz degrees).
    
    input: MCS dataframe that includes: TEMPERATURE, PRESSURE, DUST
    ,ALTITUDE, LATITUDE, LONGITUDE, H2OICE, LS, LOCALTIME
    
    output: 
    - latitude bins array
    - longitude bins array
    - a dictionary structure with each entry representing 1 LS containing a matrix of
    temperature (or dust or h2oice) quantities in each meshpoint
    '''

    #creating the base bins (won't change for the different LS's)
    
    #setting latitude bins

    Lat_bins = np.linspace(Latmin, Latmax,abs(int(((Latmax - Latmin)/latbinsize))+1))
    
    #setting longitude bins

    Long_bins = np.linspace(Longmin, Longmax,int(((Longmax - Longmin)/longbinsize))+1)

    #Setting Ls bins
    #determining LS span (minimum and maximum)
    LSmin = int(round(dfc.LS.min()))
    LSmax = int(round(dfc.LS.max()))
    #How many Ls bins total: 
    #this will be used to create array of LSs
    LSsiz  = (LSmax - LSmin)
    #creating array of Ls with spacing = binsize
    LS_arr = np.linspace(LSmin, LSmax, LSsiz+1)
    LS_arr[LS_arr!= LSmax] #removing extra entry from entry
    
    #empty dictionary for storing purposes
    results = dict() 
    
    #looping through the different LS bins to and performing the 2d binning with pressure and lat for each LS
    for i in range(int(LSsiz)):
        #pick out only entries with that specific Ls
        dfc1 = dfc[(dfc.LS >= LS_arr[i]) & (dfc.LS < LS_arr[i] + 1)]
        
        #ignore if empty
        if  dfc1.size <= 10:
            continue
        
        #binning temperature
        if choice == 'temp':
            temp = bin2d(dfc1.LATITUDE, dfc1.LONGITUDE, dfc1.TEMPERATURE, 
                         bins = [Lat_bins,Long_bins], statistic = 'mean')[0]
            ans = temp.T
            
        #binning frost temp
        elif choice == 'tfrost':
            tfrost = bin2d(dfc1.LATITUDE, dfc1.LONGITUDE, dfc1.T_frost,
                         bins = [Lat_bins,Long_bins], statistic = 'mean')[0]
            ans = tfrost.T
        
        #binning dust
        elif choice == 'dust':
            dust = bin2d(dfc1.LATITUDE, dfc1.LONGITUDE, dfc1.DUST,
                         bins = [Lat_bins,Long_bins], statistic = 'mean')[0]
            ans = dust.T
            
        #binning H2O ice
        elif choice == 'h2oice':
            h2oice = bin2d(dfc1.LATITUDE, dfc1.LONGITUDE, dfc1.H2OICE,
                                        bins = [Lat_bins,Long_bins], statistic = 'mean')[0]
            ans = h2oice.T

        #binning Pressure:
        elif choice == 'pressure':
            press = bin2d(dfc1.LATITUDE, dfc1.LONGITUDE, dfc1.PRESSURE,
                                        bins = [Lat_bins,Long_bins], statistic = 'mean')[0]
            ans = press.T
            
        #error message if no choice was made
        else: 
            print('invalid variable input please choose from the following: temp, dust, h2oice ')
           
        #save current Ls binned data to dictionary    
        results[str(LS_arr[i])] = ans
    return Lat_bins, Long_bins, results


def fill_gaps(Temps,LongV,LatV):
    '''fills in the gaps in the temperature matrix generated by long-lat binning function
    input: temperature matrix (transposed from long-lat output)
    output: filled in temperature matrix
    '''
    
    #manipulating the lat long array to align on center of pixel 
    steplon = LongV[2]- LongV[1]
    steplat = LatV[2] - LatV[1]
    Long = LongV[1:]-(0.5*steplon)
    Lat = LatV[1:]-(0.5*steplat)
    
    #subsample to remove effects of different orbits and spatial sampling patterns
    subsample = 3
    Temperatures = Temps[: , ::subsample]
    Longs = Long[::subsample]
    Lats = Lat
    
    #make pandas dataframe from the matrix with temperatures, such that coordinates are indices
    tempdf = pd.DataFrame(Temperatures).stack().rename_axis(['x', 'y']).reset_index(name='temp')
    
    #list the latitude and longitude in the dataframe by calling on those indices
    tempdf['lon'] = Longs[tempdf['y'].astype(int)]
    tempdf['lat'] = Lats[tempdf['x'].astype(int)]
    
    #create geopandas dataframe for interpolation purposes so that it knows the datapoints are in geo coordinates
    gdf = geopandas.GeoDataFrame(
        tempdf, geometry=geopandas.points_from_xy(tempdf['lon'], tempdf['lat']), crs="EPSG:4326",)
    
    #make geocube for rasterization/interpolation of missing data
    geo_grid_cubic = make_geocube(
        gdf,
        measurements=["temp"],
        resolution=(steplat, subsample*steplon),
        #interpolate_na_method="cubic",
        rasterize_function=partial(rasterize_points_griddata, method="nearest"),
    )
    
    filled_temps = ndimage.gaussian_filter(geo_grid_cubic.temp.values, sigma=1, order=0)
    lon = geo_grid_cubic.x.values
    lat = geo_grid_cubic.y.values

    
    return filled_temps, lon, lat
        
def contour_fit(cp):
    '''creates a degree-4 polynomial fit to the contour curve that is generated by matplotlib contour
    input: contour map element (note that this has to be only a one level contour at T = 170K)
    output: 2 arrays containing longitude and latitude coordinates of the fit line 
    '''
    #add all segments if it is not continuous
    ctr_pts = cp.allsegs[0]
    ctr = (np.vstack(ctr_pts))
    #sort by longitude
    sorted_ctr = sorted(ctr, key=itemgetter(0))
    sorted_ctr = np.vstack(sorted_ctr)
    cntr = sorted_ctr.T
    
    #add the first longitude reading at the end of the arrays for continuity
    longs = np.append(cntr[0], 180)
    lats = np.append(cntr[1],cntr[1][0] )
    
    longer = cntr[0] + 360
    beforer = cntr[0] - 360
    llngs = np.append(cntr[0],longer)
    llongs = np.append(beforer,llngs)
    llts = np.append(cntr[1], cntr[1])
    llats = np.append(llts, cntr[1])
    
    #fit a polynomial
    z = np.polyfit(llongs, llats, deg=30)
    f = np.poly1d(z)
    
    # calculate new x's and y's
    x_new = np.linspace(-540, 540, 1081)
    y_new = f(x_new)
    
    #make sure the beginning and end of the fit pass through the same point
    idx_init = np.where(x_new == -180)[0][0]
    idx_fin = np.where(x_new == 180)[0][0]
    fixed_pt = y_new[idx_init]
    
    idx_insert1 = np.where(llongs == 180)
    idx_insert2 = np.where(llongs == -180)
    
    llats[idx_insert1] = fixed_pt
    llats[idx_insert2] = fixed_pt
    
    w = np.ones(llongs.shape[0])
    w[idx_insert1] = 20
    w[idx_insert2] = 20
    z2 = np.polyfit(llongs,llats, deg=30, w = w)
    f2 = np.poly1d(z2)
    
    x_new2 = x_new[idx_init:idx_fin+1]
    y_new2 = f2(x_new)[idx_init:idx_fin+1]
    
    return x_new2, y_new2

def centroid_point(Longs, Lats):
    '''Calculates the centroid of the polar vortex by calculating the average of the long/lat vectors
    input: latitude and longitude array of polar vortex boundary 
    output: centroid location of polar vortax'''
    R_m = 3.3895e6 #radius mars in meters
    xs = R_m*np.cos(np.deg2rad(Lats))*np.cos(np.deg2rad(Longs))
    ys = R_m*np.cos(np.deg2rad(Lats))*np.sin(np.deg2rad(Longs))
    zs = R_m*np.sin(np.deg2rad(Lats))
    
    av_x = np.sum(xs)/len(xs)
    av_y = np.sum(ys)/len(ys)
    av_z = np.sum(zs)/len(zs)
    
    sqrrt = np.sqrt(av_x*av_x + av_y*av_y)
    
    cen_lon = np.rad2deg(np.arctan2(av_y,av_x))
    cen_lat = np.rad2deg(np.arctan2(av_z,sqrrt))
    
    return cen_lon, cen_lat



