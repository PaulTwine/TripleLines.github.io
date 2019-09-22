import ovito.io as io
import ovito.modifiers as om
import ovito.data as oid
import numpy as np
import sys
#strFilename = '/home/paul/PythonLAMMPS/70and80.eam'
strFilename = str(sys.argv[1]) #assumes ovitos PMStructure.py strFilename command line 
print(strFilename)
pipeline = io.import_file(strFilename, multiple_frames= True)
n = pipeline.source.num_frames
#print(n) #debug information
pipeline.modifiers.append(om.PolyhedralTemplateMatchingModifier(rmsd_cutoff=0.1,output_orientation = True))
pipeline.compute()
print(pipeline.output) #debug information
io.export_file(pipeline, strFilename + 'PM', "lammps_dump", frame = n, columns = ['Particle Identifier','Position.X','Position.Y', 'Position.Z', 'Velocity.X', 'Velocity.Y','Velocity.Z','c_pe1','Structure Type', 'Orientation.W','Orientation.X', 'Orientation.Y','Orientation.Z'])
