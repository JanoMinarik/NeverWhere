#include "atom.h"
#include <iostream>

/// for debugging, checking data initialization
void printAtom(atom myAtom){
  std::cout << "shells [no, ang, x, y, z]:\n";
  for(int i=0; i<myAtom.noShl; i++){
    std::cout << myAtom.atomShell[i].ang << "  ";
    std::cout << myAtom.atomShell[i].x << "  ";
    std::cout << myAtom.atomShell[i].y << "  ";
    std::cout << myAtom.atomShell[i].z << "\n";
  }

  std::cout << "contraction functions [shell no, fnc no, exp, coef]:\n";
  for(int i=0; i<myAtom.noFnc; i++){
    std::cout << myAtom.shellNumber[i] << "  " << myAtom.shellFunction[i] << "  ";
    std::cout << myAtom.exp[i] << "  " << myAtom.coef[i] << "\n";
  }

  std::cout << "===============================";
}
