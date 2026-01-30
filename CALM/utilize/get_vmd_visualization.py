import MDAnalysis as mda
import numpy as np




z_values = np.load("curvature_directory/00100_Z_fitted.npy")          
box_size = np.loadtxt("curvature_directory/dimensions.csv",delimiter=",", skiprows=1, max_rows=1, usecols=(1,2,3))  
print(box_size)

#Convert nm→Å
box_size*=10
z_values*=10

n_layers,Nx,Ny =z_values.shape 
print(z_values.shape)

dx = box_size[0] / Nx
dy = box_size[1] / Ny

x = np.linspace(-box_size[0]/2 + dx/2, box_size[0]/2 - dx/2, Nx)
y = np.linspace(-box_size[1]/2 + dy/2, box_size[1]/2 - dy/2, Ny)
X, Y = np.meshgrid(x, y, indexing="ij")


coords = []
for l in range(n_layers):
    layer_coords=np.column_stack([X.flatten(),Y.flatten(),z_values[l].flatten()])
    coords.append(layer_coords)

coords=np.vstack(coords)
n_atoms=coords.shape[0]
print(n_atoms)

resindices = np.repeat(np.arange(n_layers), Nx*Ny)  # 0,1,2
n_residues=n_layers

u=mda.Universe.empty(n_atoms=n_atoms,n_residues=n_residues,atom_resindex=resindices,trajectory=True)

for attr in ["name","resname","resid"]:
    u.add_TopologyAttr(attr)

u.atoms.names = ["C"]*n_atoms
u.residues.resnames = ["upper","middle","lower"]
u.residues.resids = [1,2,3]

u.atoms.positions=coords
u.dimensions=[*box_size, 90, 90, 90]

u.atoms.write("00100_pseudo_universe.gro")




#with mda.coordinates.XTC.XTCWriter(f"{out_dir}/fourier_curvature_fitting_{Layer}.xtc", n_atoms=n_atoms) as writer: