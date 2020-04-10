import ovito.io as io
import ovito.modifiers as om
import ovito.data as oid
import numpy as np
import sys
from ovito.vis import Viewport, TachyonRenderer
strFilename = str(sys.argv[1]) #assumes "ovitos PMStructure.py strFilename  fltRMSD "command line 
fltRMSD = float(sys.argv[2])
pipeline = io.import_file(strFilename, multiple_frames= True)
n = pipeline.source.num_frames
pipeline.add_to_scene()
pipeline.modifiers.append(om.ElasticStrainModifier(calculate_strain_tensors=False, lattice_constant = 4.05))
pipeline.modifiers.append(om.PolyhedralTemplateMatchingModifier(rmsd_cutoff=fltRMSD,output_orientation =False)) #this needs to be last otherwise the structure types are not updated
pipeline.compute()
vp = Viewport(type = Viewport.Type.Top)
vp.zoom_all()
vp.render_image(filename=strFilename + '.png', size=(320, 240),  renderer=TachyonRenderer())
#print(pipeline.output) #debug information
io.export_file(pipeline, strFilename + 'PM', "lammps_dump", frame = n, columns = ['Particle Identifier','Position.X','Position.Y', 'Position.Z', 'Velocity.X', 'Velocity.Y','Velocity.Z','c_pe1','Structure Type', 'Volumetric Strain'])
