import ovito.io as io
import ovito.modifiers as om
import ovito.data as oid
import numpy as np
import sys
#strFilename = '/home/paul/PythonLAMMPS/70and80.eam'
strFilename = str(sys.argv[1]) #assumes ovitos PMStructure.py strFilename command line 
fltCutOff = float(sys.argv[2])
pipeline = io.import_file(strFilename, multiple_frames= True)
n = pipeline.source.num_frames
pipeline.modifiers.append(om.PolyhedralTemplateMatchingModifier(rmsd_cutoff=fltCutOff,output_orientation = False))
pipeline.modifiers.append(om.ElasticStrainModifier(calculate_strain_tensors=False, lattice_constant = 4.05))
pipeline.compute()
print(pipeline.output) #debug information
io.export_file(pipeline, strFilename + 'PM', "lammps_dump", frame = n, columns = ['Particle Identifier','Position.X','Position.Y', 'Position.Z', 'Velocity.X', 'Velocity.Y','Velocity.Z','c_pe1','Structure Type', 'Volumetric Strain'])
