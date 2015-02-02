'''
Created on 02 Feb, 2015

@author: Jano Minarik

Python script for build main.cpp with following setup:
1 Neons in 100pt grid.
Neon is placed at 0-0-0 coordinates.
Points are distributed randomly.
Please check if destination of your Python compiler is same as mine.
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
            if(line[1] == ' '):
                args = line.split(' ');
            else:
                args = line.split('\t');
            if(cnt < self.noShells):
                self.ang.append(args[1]);
                self.x.append(args[2]);
                self.y.append(args[3]);
                self.z.append(args[4]);
            else:
                self.shellNo.append(args[0]);
                self.shellFnc.append(args[1]);
                self.exp.append(args[2]);
                self.coef.append(args[2]);
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
myAtom.readData("neon.txt");

# parameters for grid
gridName = 'myGrid';
gridAts = 1;
gridPts = 100;
ptRange = 3;

# myAtom.displayData();
# include libraries
main = '''#include <stdio.h>
#include <ctime>
#include "atom.h"
#include "grid.h"
\nint main(){
'''

# initialize grid
main += '  grid %s;\n' % gridName
main += '  %s.setGrid(%d, %d);\n\n' % (gridName, gridAts, gridPts)

# set atoms
for i in range(0, gridAts):
    for j in range(0, myAtom.noShells):
        main += '  %s.setAtom[%d](%d, %s, %s, %s, %s);\n' % (gridName, i, j, myAtom.ang[i], myAtom.x[i], myAtom.y[i], myAtom.z[i]);

# calculate grid and write output to file "output.txt"
main += '''
  %s.calcGrid();
  %s.printGrid2File();
''' %(gridName, gridName)
main += '  return 0;\n}'

# print main to main.cpp
print(main);
