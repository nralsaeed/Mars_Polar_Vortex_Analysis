# Mars_Polar_Vortex_Analysis

Repository of the data and scripts used in analyzing the polar vortex of Mars using MCS data

The original MCS data is obtained from NASA’s Planetary Data System (https://atmos.nmsu.edu/data_and_services/atmospheres_data/MARS/mcs.html![image](https://user-images.githubusercontent.com/40036308/168688055-24306332-0a99-44ce-9785-ad6ef7df0a3d.png) The data obtained is the observation date, ls, local time, altitude, latitude, longitude, temperature, pressure , dust opacity, and h2oice opacity for both poles, for Mars years 29 to 35.

The MCS_Vortex_modules.py file contains all the functions used to bin and manipulate the data.

The North_Polar_Vortex_Shape.ipynb and South_Polar_Vortex_Shape.ipynb files are jupyter notebooks with step by step process of binning and processing the data to characterize the shape of the polar vortex, as well as plotting routines.

The Polar_Vortex_Temp_Area.ipynb file is a jupyter notebook with step by step process of binning and processing the data to find the area and average temperature of the polar vortex, as well as plotting routines.

The Shape_Size_Temp_Dust_Plots.ipynb file describes the plotting routine for the seasonal variability in vortex area, vortex temperature and global maps of dust opacity.

The results of the data processing and binning routines are stored in the Results folder which can then be used for the pplotting routines, below is a description of folder within the results:
'Snow Cloud Density Distribution' contains files representing the cloud density distribution in longitude and latitude for each year and both south and north
'Vortex Areas' contains files with dataframes listing the vortex areas as a function of Ls, one file for each the north and south
'Vortex Temperatures' contains files with dataframes listing the average vortex temperatures as a function of Ls, one file for each the north and south
'Vortex Contours' contains files storing the contours/boundary of the vortex, one file for each year split into the north and south
