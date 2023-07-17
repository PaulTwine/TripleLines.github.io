#%%
import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from mpl_toolkits.mplot3d import proj3d
# %%
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{bm}')
plt.rcParams['figure.dpi'] = 300
strTime = 'Time in fs'
strPotentialEnergy = 'Potential energy in eV'
strVolume =r'Volume in \AA$^{3}$'
#%%
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

#plt.rcParams.update({'font.size': 15})
#%%
#plt.annotate(text='$r_0$', xy=(0.4,-1),ha='center',fontsize=24,c='black')
#plt.plot([0.025,2],[0.025,3])
#plt.annotate(text='', xy=(0,-0.5), xytext=(2,-3),c='black',fontsize=5, arrowprops={'arrowstyle':'<->'},color='black')
#plt.plot([0,-1],[0,1])

#%matplotlib qt
plt.quiver(0,0,1,3, angles='xy', scale_units='xy', scale=1)
plt.annotate(r'$\bm{\gamma}_{\textrm{A,B}}$',xy=(1,3),xytext=(1,3))
plt.quiver(0,0,-2,1,angles='xy', scale_units='xy', scale=1)
plt.annotate(r'$\bm{\gamma}_{\textrm{B,C}}$',xy=(-2,1),xytext=(-2,1.25))
plt.quiver(0,0,1,-4,angles='xy', scale_units='xy', scale=1)
plt.annotate(r'$\bm{\gamma}_{\textrm{A,C}}$',xy=(1,-4),xytext=(1,-4))
plt.annotate('A', xy=(1,0))
plt.annotate('B', xy=(-0.5,1))
plt.annotate('C', xy=(-0.9,-0.9))

plt.quiver(5,3,1,-4,width=0.005,angles='xy', scale_units='xy', scale=1,linestyle='--',color='black')
plt.quiver(6,-1,-2,1,width=0.005,angles='xy', scale_units='xy', scale=1,linestyle='--',color='black')
plt.quiver(4,0,1,3,width=0.005,angles='xy', scale_units='xy', scale=1,linestyle='--',color='black')

plt.axis('off')
plt.scatter([0], [0], s=2*4**2*np.pi,c='white',linewidths=1.5)
plt.scatter([0], [0], s=2*4**2*np.pi,facecolors='', edgecolors='dimgrey',linewidths=1.5)
plt.xlim([-3,8])
plt.ylim([-5,4])
plt.scatter([0], [0], s=5,marker='o',c='dimgrey')
plt.show()
# %%
#%matplotlib qt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=20., azim=-90)


z = 0#-1.5 
a = Arrow3D([0, 1], [0, 3], [0, z], mutation_scale=20, lw=2, arrowstyle="->", color="black")
b = Arrow3D([0, -2], [0, 1], [0, z], mutation_scale=20, lw=2, arrowstyle="->", color="black")
c = Arrow3D([0, 1], [0, -4], [0, z], mutation_scale=20, lw=2, arrowstyle="->", color="black")
d = Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=20, lw=2, arrowstyle="->", color="dimgrey")
ax.add_artist(a)
ax.add_artist(b)
ax.add_artist(c)
ax.add_artist(d)
# gf.EqualAxis3D(ax)

ax.set_xlim([-2,1])
ax.set_ylim([-4,3])
ax.set_zlim([-2,1])
ax.scatter(0,0,0,c='black',s=4)
ax.text(1,3,z,r'$\bm{\gamma}_{\textrm{A,B}}$')
ax.text(-2.1,-0.5,z,r'$\bm{\gamma}_{\textrm{B,C}}$')
ax.text(1,-4,z,r'$\bm{\gamma}_{\textrm{A,C}}$')
ax.text(0,0,1,r'$\bm{l}_{\textrm{A,B,C}}$')
#gf.EqualAxis3D(ax)
plt.tight_layout()
plt.axis('off')
plt.show()
#plt.annotate('',xy=(-3,2),xytext=(0,0),arrowprops={'arrowstyle':'->'})
#plt.axes('none')

# plt.axis('off')
# plt.scatter([0], [0], s=2*4**2*np.pi,facecolors='', edgecolors='black',linewidths=1.5)
# plt.scatter([0], [0], s=2*18**2*np.pi,facecolors='', edgecolors='black', linestyle ='dashed',alpha=0)

# %%
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=20., azim=-90)

#ax.plot([0,1],[0,0],[0,0])
#ax.plot([0,-1],[0,0],[0,0])

z = 0#-1.5 
a = Arrow3D([0, 1], [0, 3], [0, z], mutation_scale=20, lw=2, arrowstyle="->", color="black")
b = Arrow3D([0, -2], [0, 1], [0, z], mutation_scale=20, lw=2, arrowstyle="->", color="black")
c = Arrow3D([0, 1], [0, -4], [0, z], mutation_scale=20, lw=2, arrowstyle="->", color="black")
d = Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=20, lw=2, arrowstyle="->", color="dimgrey")
ax.add_artist(a)
ax.add_artist(b)
ax.add_artist(c)
ax.add_artist(d)
# gf.EqualAxis3D(ax)

ax.set_xlim([-2,1])
ax.set_ylim([-4,3])
ax.set_zlim([-2,1])
ax.scatter(0,0,0,c='black',s=4)
ax.text(1,3,z,r'$\bm{\gamma}_{\textrm{A,B}}$')
ax.text(-2.1,-0.5,z,r'$\bm{\gamma}_{\textrm{B,C}}$')
ax.text(1,-4,z,r'$\bm{\gamma}_{\textrm{A,C}}$')
ax.text(0,0,1,r'$\bm{l}_{\textrm{A,B,C}}$')
#gf.EqualAxis3D(ax)
plt.tight_layout()
plt.axis('off')
plt.show()
#plt.annotate('',xy=(-3,2),xytext=(0,0),arrowprops=
# %%
#ax = plt.gca()
#ax.set_aspect('equal', adjustable='box')
r = 1.5
d = 0.45
tIR = np.linspace(0,d,50)
tOR = np.linspace(d,0.75,50)
tIL = np.linspace(-d,0,50)
tOL = np.linspace(-0.75,-d,50)
tI = np.linspace(-d,d,100)
plt.plot(tI,-tI**2/r,c='black', linestyle='dashed')
#plt.plot(tIL,-tIL**2/2,c='black', linestyle='dashed')
plt.plot(tOR,-tOR**2/r,c='black')
plt.plot(tOL,-tOL**2/r,c='black')
plt.plot([-d,0],[-d**2/r,0.5*d*np.sqrt(3)/r-d**2/r],c='black')
plt.plot([d,0],[-d**2/r,0.5*d*np.sqrt(3)/r-d**2/r],c='black')
plt.plot([0,-d/2],[0.5*d*np.sqrt(3)/r-d**2/r,0.75*d*np.sqrt(3)/r-d**2/r],c='black', linestyle='dotted')
plt.plot([0,0],[1.5*d*np.sqrt(3)/r-d**2/r,0.5*d*np.sqrt(3)/r-d**2/r],c='black')
plt.annotate(r'$\pi/3$',xy=(-0.2,0.1),fontsize=15)
plt.axes().set_aspect('equal')
plt.axis('equal')
#plt.plot(tI,np.sqrt(d**2-tI**2))
#plt.plot(tI,-np.sqrt(d**2-tI**2))
#plt.axis('off')
plt.show()
# %%
%matplotlib qt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=20., azim=-90)


z = 0#-1.5 
#a = Arrow3D([0, 1], [0, 3], [0, z], mutation_scale=20, lw=2, arrowstyle="->", color="black")
b = Arrow3D([0, -1], [0, 0], [0, 4], mutation_scale=20, lw=2, arrowstyle="->", color="black")
c = Arrow3D([0, -1], [0, 0], [4, 4], mutation_scale=20, lw=2, arrowstyle="->", color="black")
d = Arrow3D([0, 0], [0, 0], [0, 4], mutation_scale=20, lw=2, arrowstyle="->", color="black")
#ax.add_artist(a)
ax.add_artist(b)
ax.add_artist(c)
ax.add_artist(d)
# gf.EqualAxis3D(ax)

ax.set_xlim([-2,1])
ax.set_ylim([-4,3])
ax.set_zlim([-2,1])
ax.scatter(0,0,0,c='black',s=4)
ax.text(-0.85,0,2,r'$\bm{\xi}$',fontsize=15)
ax.text(-0.25,0,1.25,r'$\theta$',fontsize=15)
ax.text(0.125,0,2,r'$\bm{\xi}_{n}$',fontsize=15)
ax.text(-0.5,0,4.25,r'$\bm{\xi}_{t}$',fontsize=15)
ax.plot([-1.25,1.25],[-1.25,-1.25],[0,0],c='black', linestyle ='dashed')
ax.plot([1.25,1.25],[-1.25,1.25],[0,0],c='black', linestyle ='dashed')
ax.plot([1.25,-1.25],[1.25,1.25],[0,0],c='black', linestyle ='dashed')
ax.plot([-1.25,-1.25],[1.25,-1.25],[0,0],c='black', linestyle ='dashed')
ax.plot([0,-1],[0,0],[0,0],c='black', linestyle ='dotted')
ax.plot([-1,-1],[0,0],[0,4],c='black', linestyle ='dotted')



gf.EqualAxis3D(ax)
plt.tight_layout()
plt.axis('off')
plt.show()

# %%
