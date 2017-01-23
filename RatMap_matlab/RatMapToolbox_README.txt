RatMap Toolbox 
==============

1. Associated publication:
--------------------------

“The Morphology of the Rat Vibrissal Array: A Model for Quantifying Spatiotemporal Patterns of Whisker-Object Contact”
Towal RB*, Quist BW*, Gopal V, Solomon JH, and Hartmann MJZ
PLoS Computational Biology
* Authors contributed equally

2. Toolbox Contents (4 files)
-----------------------------

a. RatMapToolbox_README.txt 	~ Text readme file
b. Plot_RatMap.m		~ MATLAB script to plot the complete RatMap in 3D
c. Get_RatMap.m 		~ MATLAB function to compute the RatMap whiskers
d. Get_RatMapEllipsoid.m	~ MATLAB function to compute the RatMap mystacial ellipsoids

3. Getting started
------------------

Place the unzipped files within the same directory. 
In the command line, type:

	Plot_RatMap

You should now see the complete RatMap plotted in 3D.
Whisker 3D positions and orientations are computed using Get_RatMap.
Mystacial pad ellipsoids are computed using Get_RatMapEllipsoid.

4. Customizing the RatMap
-------------------------

The “Get_RatMap” and “Get_RatMapEllipsoid” functions provide access to all parameters used to generate the RatMap. 

For “Get_RatMap,” any subset of the full array of whiskers can be selected using the first input, while all subsequent inputs to the function require a text string with the parameter name and the parameter itself (following the structure of a typical MATLAB function). Parameters use a function dependent on row/column position in the array or can be set directly.

Example function calls to “Get_RatMap” include:

% Generate both L&R full arrays
[x,y,z] = Get_RatMap();    

% Generate Left full array		
[x,y,z] = Get_RatMap('L');  

% Generate Left A-row		
[x,y,z] = Get_RatMap('LA');  	

% Generate Left A1 whisker	
[x,y,z] = Get_RatMap('LA1'); 	

% Generate full array where whiskers have 50 data points each	
[x,y,z] = Get_RatMap(‘’,'Npts',50); 

% Generate Left full array with new function for theta that depends on row and column	
[x,y,z] = Get_RatMap('L','EQ_W_th',[9.9 9.9 99.9]);
                                			
% Generate LA1 whisker with a projection theta angle set to 90 degrees
[x,y,z] = Get_RatMap('LA1','W_th',90);