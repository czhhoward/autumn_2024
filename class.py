
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy.io import fits
from dataclasses import dataclass,field

@dataclass
class starmovementplot:
    file_path: str
    hdu_index: int
    hdul: fits.HDUList = field(init=False)  # 使用 `field(init=False)` 表示该数值不在init中赋值
    hdu: np.ndarray = field(init=False)
    header: np.ndarray = field(init=False)
    data: np.ndarray = field(init=False)
    unit: str = field(init=False)
    def __post_init__(self):
        self.hdul = fits.open(self.file_path)
        self.hdu,self.header,self.data,self.unit=self.get_value()
        self.hdul.close()
    def get_value(self):
      hdu_array=np.recarray(len(self.hdul),dtype=[('index', 'i4'),('HDU_name',"U20")])

      for i, hdu in enumerate(self.hdul):
        hdu_array[i][0]=i
        hdu_array[i][1]=hdu.name

      head=self.hdul[self.hdu_index].header
      header_array=np.recarray(len(head),dtype=[('name','U10'),('value','U10')])
      for k, (key, value) in enumerate(head.items()):
        header_array[k][0]=key
        header_array[k][1]=value
        
      if self.hdul[self.hdu_index].data is not None:
        data_array = self.hdul[self.hdu_index].data


      unit=''
      unit_list=['ra','dec','glob','centroid']
      check_header=[i.value for i in header_array]

      for k in check_header:
        if 'ra' in k.lower() or 'dec' in k.lower():
          unit = 'ra/dec'


      if 'glob' in hdu_array[self.hdu_index]['HDU_name'].lower() or 'centroid' in hdu_array[self.hdu_index]['HDU_name'].lower():
        unit='milliarcsecond'

      return hdu_array,header_array,data_array,unit

    def data_process(self,index_x,index_y,transpose=False,frame=None,field=False):
      if field:
        x_coords,y_coords=self.data.field(index_x),self.data.field(index_y)
        if transpose:
           x_coords, y_coords=x_coords.T,y_coords.T
      else:
        x_coords,y_coords=np.array(self.data[index_x]),np.array(self.data[index_y])
        if transpose:
           x_coords, y_coords=x_coords.T,y_coords.T

      if frame == None:
        frame=[i for i in range((len(x_coords)))]
      return frame,x_coords,y_coords

    def unit_conversion(self):
      if self.unit == "milliarcsecond":
        base=3600000
      return int(base)
   

    def update_animation(self,frame,x_coord,y_coord,scatter,frame_text=None,var_text=None,multiplier=10000): #在调用的时候不需要输入frame，但是在写函数的时候frame要在第一个（self之后）
      x_coords=(x_coord[frame]-x_coord[0])*multiplier+x_coord[0]
      y_coords=(y_coord[frame]-y_coord[0])*multiplier+y_coord[0]
      try:
        frame_text.set_text(f'Frame: {frame}')
        var_x = x_coords.var()
        var_y = y_coords.var()
        var_text.set_text(f'Var: ({var_x:.3f}, {var_y:.3f})')
        scatter.set_offsets(np.c_[x_coords, y_coords])
        return [scatter, frame_text,var_text]
      except AttributeError:
        scatter.set_offsets(np.c_[x_coords, y_coords])
        return scatter

@dataclass
class averagemovementplot(starmovementplot):


  def plot(self, index_x, index_y,frame=None,transpose=False,field=False,figsizes=(12,8),set_x_lim=False,set_y_lim=False):
    frame,x_coords,y_coords=self.data_process(index_x, index_y,frame=frame,transpose=transpose,field=field)

    fig = plt.figure(figsize=(figsizes))
  
    x_coord=[abs(i.mean())for i in x_coords]
    y_coord=[abs(i.mean())for i in y_coords]
    avg_dis=[abs((x_coords[i].mean())**2+abs(y_coords[i].mean())**2)**0.5 for i in range(len(frame))]

    ax1 = fig.add_subplot(1,3,1)
    plot1=ax1.plot(frame,x_coord)
    plt.title('x_avg movement')
    plt.ylabel('centroid x')
    plt.xlabel('frame')
    plt.grid()

    ax2=fig.add_subplot(1,3,2)
    plot2=ax2.plot(frame,y_coord)
    plt.title('y_avg movement')
    plt.ylabel('centroid y')
    plt.xlabel('frame')
    plt.grid()


    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(frame, avg_dis)
    plt.title('Avg Distance Movement')
    plt.ylabel('Avg Distance')
    plt.xlabel('Frame')
    plt.grid()

    if set_x_lim:
      ax1.set_xlim(set_x_lim)
      ax2.set_xlim(set_x_lim)
      ax3.set_xlim(set_x_lim)
    if set_y_lim:
      ax1.set_ylim(set_y_lim)
      ax2.set_ylim(set_y_lim)
      ax3.set_ylim(set_y_lim)
    plt.show()

@dataclass
class histogram_movement(starmovementplot):

  def plot(self, index_x,index_y,num_objects,frame=None,field=False,transpose=False,bins=10,alpha=0.7,type_rarray=[('x_range', 'f8'), ('y_range', 'f8'), ('index', 'i4')]):
    frame,x_coord,y_coord=self.data_process( index_x, index_y,frame=frame,field=field,transpose=transpose)

    empty_arr=np.recarray(num_objects,dtype=type_rarray)
    empty_arr[f'{type_rarray[0][0]}']=[x_coord[i].max() - x_coord[i].min() for i in range(num_objects)]
    empty_arr[f'{type_rarray[1][0]}']=[y_coord[i].max() - y_coord[i].min() for i in range(num_objects)]
    empty_arr[f'{type_rarray[2][0]}'] = [i for i in range(num_objects)]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(empty_arr[f'{type_rarray[0][0]}'], bins=bins, color='blue', alpha=alpha)
    plt.title('Distribution of x_range')
    plt.xlabel('x_range')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(empty_arr[f'{type_rarray[1][0]}'], bins=bins, color='orange', alpha=alpha)
    plt.title('Distribution of y_range')
    plt.xlabel('y_range')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

@dataclass
class movement_animation(starmovementplot):

  def plot(self, index_x,index_y,frame=None,field=False,transpose=False,frame_text=False,var_text=False):
    frame,x_coord,y_coord=self.data_process(index_x, index_y,frame=frame,field=field,transpose=transpose)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    first_F_x=x_coord[0]
    first_F_y=y_coord[0]
    scatter=ax.scatter(first_F_x,first_F_y)

    if frame_text:
      frame_text = fig.text(0.5, 0.95, '', ha='center',fontsize=12,color='black')
    if var_text:
      var_text = fig.text(0.5, 0.9, '', ha='center',fontsize=12,color='black')

    plt.xlabel('centroid x')
    plt.ylabel('centroid y')

    ani = FuncAnimation(fig, self.update_animation,fargs=(x_coord, y_coord, scatter,frame_text,var_text), frames=len(frame), interval=1,blit=False,repeat=False)
    plt.grid()
    plt.show()


# a=movement_animation('brightestnonsat100_rot.fits',4)
# a.plot(0,1,field=True,transpose=True,frame_text=True,var_text=True)

# b=histogram_movement('brightestnonsat100_rot.fits',4)
# b.plot(0,1,89,field=True)

# c=averagemovementplot('brightestnonsat100_rot.fits',5)

# c.plot(0,1,field=True,transpose=True,set_y_lim=(0,1.5e-13))


#avergae raw-residuaol vs real time
#average(raw) - average(residual)

#unit determination
#read data