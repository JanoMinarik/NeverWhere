'''
Created on 02 Feb, 2015

@author: Jano Minarik

Scenario:
one Neon atom in grid of 101 points distributed equidistantly
in cube of 8x8x8 atomic units.
insert destination of your compiler below:
'''
#!/usr/bin/python

#import argparse

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


# Experiment Data hardcoed           
myAtom = atom(6, 25)
myAtom.readData("./input/neon.txt")
myAtom.readDensityMatrix("./input/dmat.txt")
gridName = 'myGrid'
gridPts = 16496
ptStart = 0
ptEnd = 10
noShl = len(myAtom.ang)
noFnc = len(myAtom.coef)

# Experiment Data from Parser
#parser = argparse.ArgumentParser(description='Generates main.cpp for example scenario.')
#parser.add_argument('-i', '--input', dest='ifile', default='./input/neon.txt', help='Specify input file with your atom coordinates and contration functions.')
#parser.add_argument('-o', '--output', dest='ofile', default='output.txt', help='Specify output file.')
#parser.add_argument('-g', '--grid', dest='gfile', default='None', help='Specify input file with coordinates of gridpoint and its weight.')
#parser.add_argument('-d', '--density', dest='dfile', default='./input/dmat.txt', help='Specify file with density matrix for your atom.')

# myAtom.displayData();
# myAtom.displayDensityMatrix();
# include libraries
main = '''#include "atom.h"
#include "grid.h"
\nint main(){
'''

# initialize grid
main += '  grid %s;\n' % gridName

# set atom
main += '  int shellNos [] = {'
for i in range(0, len(myAtom.shellNo)):
    main += '%s, ' % (myAtom.shellNo[i])
main += '};\n'
main += '  int shellFncs [] = {'
for i in range(0, len(myAtom.shellFnc)):
    main += '%s, ' % (myAtom.shellFnc[i])
main += '};\n'
main += '  double shellCoefs [] = {'
for i in range(0, len(myAtom.coef)):
    main += '%s, ' % (myAtom.coef[i])
main += '};\n'
main += '  double shellExps [] = {'
for i in range(0, len(myAtom.exp)):
    main += '%s, ' % (myAtom.exp[i])
main += '};\n\n'

main += '  double dMat[%d][%d] = {\n' %( myAtom.noFnc, myAtom.noFnc)
for i in range(myAtom.noFnc):
    main += '{'
    for j in range(myAtom.noFnc):
        main += '%f, ' % (myAtom.densMat[i][j])
    main += '},\n' 
main += '  };\n\n'
main += '  double *pDMat = &dMat[0][0];\n'

main += '  %s.initAtom(%s, %s);\n' % (gridName, noShl, noFnc)
main += '  %s.setGrid(%d);\n' %(gridName, gridPts)
main += '  %s.setShellNumber(shellNos);\n' % (gridName)
main += '  %s.setShellFunction(shellFncs);\n' % (gridName)
main += '  %s.setExp(shellExps);\n' % (gridName)
main += '  %s.setCoef(shellCoefs);\n' % (gridName)
main += '  %s.setDensityMatrix(pDMat);\n' %(gridName)

# set atom shells
for i in range(0, noShl):
    main += '  %s.setShell(%d, %s, %s, %s, %s);\n' % (gridName ,i, myAtom.ang[i], myAtom.x[i], myAtom.y[i], myAtom.z[i])

# set points to equidistant range
#main += '\n  %s.setCoord(%d, %d);\n' % (gridName, ptStart, ptEnd)

# set points from file
main += '\n  %s.setCoordFile((char*)"./input/grid.txt");' % (gridName)
# calculate grid and write output to file "output.txt"
main += '''
  %s.calcGrid();
  %s.printGridInfo();
  %s.printGrid();
''' %(gridName, gridName, gridName)
main += '  return 0;\n}'

# print main to main.cpp
print(main);
