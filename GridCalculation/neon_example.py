'''
Created on 02 Feb, 2015

@author: Jano Minarik

Scenario:
one Neon atom in grid of 101 points distributed equidistantly
in cube of 8x8x8 atomic units.
insert destination of your compiler below:
'''
#!/usr/bin/python

class atom:
    def __init__(self, noShells=0, noFunctions=0):
        self.noShells=noShells
        self.noFunctions=noFunctions
        self.ang = []
        self.x = []
        self.y = []
        self.z = []
        self.shellNo = []
        self.shellFnc = []
        self.exp = []
        self.coef = []
    
    def readData(self, fileName):
        f = open(fileName, "r")
        cnt = 0
        while(cnt < (self.noShells+self.noFunctions)):
            line = f.readline()
            line = line.lstrip()
            line = line.replace('\n','')
            if(len(line) < 1):
                continue
            if(not self.representsInt(line[0])):
                continue
            args = line.split(' ')
	    if(cnt < self.noShells):
                self.ang.append(int(args[1]))
                self.x.append(float(args[2]))
                self.y.append(float(args[3]))
                self.z.append(float(args[4]))
            else:
                self.shellNo.append(int(args[0]))
                self.shellFnc.append(int(args[1]))
                self.exp.append(float(args[2]))
                self.coef.append(float(args[3]))
            cnt += 1
    
    def readDensityMatrix(self, fileName):
        if self.ang == None or len(self.ang) == 0:
           print("Initialize data first.")
           return 1
        
        self.noFnc = 0
        for a in self.ang:
          if a == 0:
            self.noFnc += 1
          if a == 1:
            self.noFnc += 3
          if a == 2:
            self.noFnc += 6
          if a == 3:
            self.noFnc += 10
        self.densMat = [[0 for x in range(self.noFnc)] for x in range(self.noFnc) ]
        with open(fileName) as openfileobject:
          for line in openfileobject:
            if(len(line)<1):
               continue;
            if(line[0] == '#'):
               continue;
            args = line.split(' ')
            self.densMat[int(args[0])][int(args[1])] = float(args[2])

    def displayData(self):
        print('Shells: ', self.ang)
        print('x: ', self.x)
        print('y: ', self.y)
        print('z: ', self.z)
        print('shell No.: ', self.shellNo)
        print('function No.: ', self.shellFnc)
        print('exponents: ', self.exp)
        print('coeficients: ', self.coef)
        
    def displayDensityMatrix(self):
        for x in range(self.noFnc):
          for y in range(self.noFnc):
            print("%d %d %f" %(x, y, self.densMat[x][y]))

    def representsInt(self,s):
        try: 
            int(s)
            return True
        except ValueError:
            return False

def cntLines(name):
    with open(name) as f:
       return sum(1 for _ in f)

# Experiment Parameters           
myAtom = atom(6, 25)
myAtom.readData("./input/neon-dz/basis.txt")
myAtom.readDensityMatrix("./input/neon-dz/dmat.txt")
gridFile = './input/neon-dz/grid.txt'
gridName = 'myGrid'
gridPts = cntLines(gridFile); 
ptStart = 0
ptEnd = 10
noFnc = len(myAtom.coef)
noShl = len(myAtom.ang)
precision = 1e-12

# myAtom.displayData();
# myAtom.displayDensityMatrix();

def main():
# include libraries
  build = '''#include "atom.h"
#include "grid.h"
  \nstatic double precision = %s
  \nint main(){
  ''' % (precision)

# initialize grid
  build += '  grid %s;\n' % gridName

# set atom
  build += '  int shellNos [] = {'
  for i in range(0, len(myAtom.shellNo)):
      build += '%s, ' % (myAtom.shellNo[i])
  build += '};\n'
  build += '  int shellFncs [] = {'
  for i in range(0, len(myAtom.shellFnc)):
      build += '%s, ' % (myAtom.shellFnc[i])
  build += '};\n'
  build += '  double shellCoefs [] = {'
  for i in range(0, len(myAtom.coef)):
      build += '%s, ' % (myAtom.coef[i])
  build += '};\n'
  build += '  double shellExps [] = {'
  for i in range(0, len(myAtom.exp)):
      build += '%s, ' % (myAtom.exp[i])
  build += '};\n\n'

  build += '  double dMat[%d][%d] = {\n' %( myAtom.noFnc, myAtom.noFnc)
  for i in range(myAtom.noFnc):
      build += '{'
      for j in range(myAtom.noFnc):
          build += '%f, ' % (myAtom.densMat[i][j])
      build += '},\n' 
  build += '  };\n\n'
  build += '  double *pDMat = &dMat[0][0];\n'

  build += '  %s.initAtom(%s, %s);\n' % (gridName, noShl, noFnc)
  build += '  %s.setGrid(%d);\n' %(gridName, gridPts)
  build += '  %s.setShellNumber(shellNos);\n' % (gridName)
  build += '  %s.setShellFunction(shellFncs);\n' % (gridName)
  build += '  %s.setExp(shellExps);\n' % (gridName)
  build += '  %s.setCoef(shellCoefs);\n' % (gridName)
  build += '  %s.setDensityMatrix(pDMat);\n' %(gridName)

# set atom shells
  for i in range(0, noShl):
      build += '  %s.setShell(%d, %s, %s, %s, %s);\n' % (gridName ,i, myAtom.ang[i], myAtom.x[i], myAtom.y[i], myAtom.z[i])


# set points from file
  build += '\n  %s.setCoordFile((char*)"%s");' % (gridName, gridFile)

# set points to equidistant range
  #build += '\n  %s.setCoord(%d, %d);\n' % (gridName, ptStart, ptEnd)

# calculate grid and write output to file "output.txt"
  build += '''
  %s.calcGrid();
  %s.printGridInfo();
  %s.printGrid();
  ''' %(gridName, gridName, gridName)
  build += '  return 0;\n}'

# print build to build.cpp
  print(build)

if __name__ == '__main__':
  main()
