# autumn_2024
Dataclass for plotting the position of Kepler data (can be in either pixel coordiantes or RA and DEC).<br />
It enables 3 graphs/animations:
- averagemovementplot: plots the average movement of all stars per frame in both x,y directions
- histogram_movement: plots the histogram about the difference of movement of each star in all frame per stars
- movement_animation:plots the animation about the movements of all stars
and the input file should be in FITS format.<br />
> You don't have to preprocess the file to extract x_coord and y_coord, but rather this dataclass do that for you. You want to know if you need to transpose/field (see below) for the data so that the final array for plotting is in the shape of (#of time frames * the coordinate each object)
### The General Parameters of Three Plotting Functions

""""

Args:<br />
- index_x and index_y: the index of x coordinate and y coordinate (column)
- frame: numbers of frames. If not given, then assumes the length of x_coord. Default: None
- transpose: Transpose of dara array (x*y) ->(y*x). Default: False
- field:Take the column of the file as x/y coordinate. If Default, then take the row of the file Default:False

""""

### Additional parameters 

averagemovementplot
- figsizes: the size of the 
