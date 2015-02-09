'''
Created on 02 Feb, 2015

@author: Jano Minarik

Scenario:
one Neon atom in grid of 100 points distributed equaly
in cube of 8x8x8 atomic units.
insert destination of your compiler below:
'''
#!/usr/bin/python

class atom:
    def __init__(self, noShells=0, noFunctions=0):
        self.noShells=noShells;
        self.noFunctions=noFunctions;
        self.ang = [];
        self.x = [];
        self.y = [];
        self.z = [];
        self.shellNo = [];
        self.shellFnc = [];
        self.exp = [];
        self.coef = [];
    
    def readData(self, fileName):
        f = open(fileName, "r");
        cnt = 0;
        while(cnt < (self.noShells+self.noFunctions)):
            line = f.readline();
            line = line.lstrip();
            line = line.replace('\n','');
            if(len(line) < 1):
                continue;
            if(not self.representsInt(line[0])):
                continue;
            args = line.split(' ');
	    if(cnt < self.noShells):
                self.ang.append(int(args[1]));
                self.x.append(float(args[2]));
                self.y.append(float(args[3]));
                self.z.append(float(args[4]));
            else:
                self.shellNo.append(int(args[0]));
                self.shellFnc.append(int(args[1]));
                self.exp.append(float(args[2]));
                self.coef.append(float(args[2]));
            cnt += 1;
            
    def displayData(self):
        print('Shells: ', self.ang);
        print('x: ', self.x);
        print('y: ', self.y);
        print('z: ', self.z);
        print('shell No.: ', self.shellNo);
        print('function No.: ', self.shellFnc);
        print('exponents: ', self.exp);
        print('coeficients: ', self.coef);
        
    def representsInt(self,s):
        try: 
            int(s)
            return True
        except ValueError:
            return False
            
myAtom = atom(6, 25);
myAtom.readData("./input/neon.txt");
gridName = 'myGrid';
gridPts = 100;
ptStart = 0;
ptEnd = 8;
noShl = len(myAtom.ang);
noFnc = len(myAtom.coef);

# myAtom.displayData();
# include libraries
main = '''#include "atom.h"
#include "grid.h"
\nint main(){
'''

# initialize grid
main += '  grid %s;\n' % gridName
main += '  %s.setGrid(%d);\n\n' % (gridName, gridPts)

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

main += '  %s.initAtom(%s, %s);\n' % (gridName, noShl, noFnc)
main += '  %s.setShellNumber(shellNos);\n' % (gridName)
main += '  %s.setShellFunction(shellFncs);\n' % (gridName)
main += '  %s.setExp(shellCoefs);\n' % (gridName)
main += '  %s.setCoef(shellExps);\n' % (gridName)

# set atom shells
for i in range(0, noShl):
    main += '  %s.setShell(%d, %s, %s, %s, %s);\n' % (gridName ,i, myAtom.ang[i], myAtom.x[i], myAtom.y[i], myAtom.z[i])

# set points to equidistant range
main += '\n  %s.setCoord(%d, %d);\n' % (gridName, ptStart, ptEnd)

# calculate grid and write output to file "output.txt"
main += '''
  %s.calcGrid();
  %s.printGrid();
''' %(gridName, gridName)
main += '  return 0;\n}'

# print main to main.cpp
print(main);
