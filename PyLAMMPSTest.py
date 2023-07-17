import lammps

objLammps = lammps.PyLammps()
objLammps.file('TemplateMin.in')
objLammps.command('out.dat')
objLammps.close()



