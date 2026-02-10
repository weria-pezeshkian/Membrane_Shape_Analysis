#import MDAnalysis as mda
#import numpy as np
#
#
#z_values=np.load("curvature_directory/01000_Z_fitted.npy")          
#box_size=np.loadtxt("curvature_directory/dimensions.csv",delimiter=",", skiprows=1, max_rows=1, usecols=(1,2,3))  
#
#Convert nm→Å
#box_size=box_size*10
#z_values=z_values*10
#
#print(box_size)
#
#n_layers,Nx,Ny =z_values.shape 
#print(z_values.shape)
#
#dx = box_size[0] / Nx
#dy = box_size[1] / Ny
#
#x = -box_size[0]/2 + (np.arange(Nx) + 0.5) * dx
#y = -box_size[1]/2 + (np.arange(Ny) + 0.5) * dy
#
#X, Y = np.meshgrid(x, y, indexing="ij")
#
#print(x[0] - dx/2, x[-1] + dx/2)
#print(y[0] - dy/2, y[-1] + dy/2)
#
#
#coords = []
#for l in range(n_layers):
#    layer_coords=np.column_stack([X.flatten(),Y.flatten(),z_values[l].flatten()])
#    coords.append(layer_coords)
#
#coords=np.vstack(coords)
#n_atoms=coords.shape[0]
#print(n_atoms)
#
#resindices = np.repeat(np.arange(n_layers), Nx*Ny)  # 0,1,2
#n_residues=n_layers
#
#u=mda.Universe.empty(n_atoms=n_atoms,n_residues=n_residues,atom_resindex=resindices,trajectory=True)
#
#for attr in ["name","resname","resid"]:
#    u.add_TopologyAttr(attr)
#
#u.atoms.names = ["C"]*n_atoms
#u.residues.resnames = ["upper","middle","lower"]
#u.residues.resids = [1,2,3]
#
#u.atoms.positions=coords
#u.dimensions=[*box_size, 90, 90, 90]
#
#u.atoms.write("01000_pseudo_universe.gro")




#with mda.coordinates.XTC.XTCWriter(f"{out_dir}/fourier_curvature_fitting_{Layer}.xtc", n_atoms=n_atoms) as writer:



import MDAnalysis as mda
import numpy as np

# Load Z-values (already in nm)
z_values = np.load("curvature_directory/01000_Z_fitted.npy")  
z_values=z_values*10

# Load box size in nm
box_size = np.loadtxt("curvature_directory/dimensions.csv", delimiter=",", skiprows=1, max_rows=1, usecols=(1,2,3))  

n_layers, Nx, Ny = z_values.shape

# X/Y grid (nm)
x = np.linspace(0, box_size[0], Nx, endpoint=False)
y = np.linspace(0, box_size[1], Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

# Reorder layers: upper, middle, lower
z_values_correct_order = np.stack([
    z_values[0],  # Upper
    z_values[2],  # Middle
    z_values[1],  # Lower
], axis=0)

coords = np.vstack([
    np.column_stack([X.flatten(), Y.flatten(), z_values_correct_order[l].flatten()])
    for l in range(n_layers)
])

# Shift Z to center
Lz = box_size[2]
z_mid = (coords[:,2].min() + coords[:,2].max()) / 2
coords[:,2] += Lz/2 - z_mid

# Build pseudo-universe
resindices = np.repeat(np.arange(n_layers), Nx*Ny)
u = mda.Universe.empty(n_atoms=coords.shape[0], n_residues=n_layers,
                        atom_resindex=resindices, trajectory=True)
for attr in ["name", "resname", "resid"]:
    u.add_TopologyAttr(attr)

u.atoms.names = ["C"]*coords.shape[0]
u.residues.resnames = ["upper", "middle", "lower"]
u.residues.resids = [1, 2, 3]

u.atoms.positions = coords  # in nm
u.dimensions = [*box_size, 90.0, 90.0, 90.0]  # nm

u.atoms.write("01000_pseudo_universe.gro")


