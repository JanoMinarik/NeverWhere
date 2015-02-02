#include "atom.h"

void initAtom(atom myAtom, int noShl, int noFnc){
  myAtom.noShl = noShl;
  myAtom.noFnc = noFnc
  myAtom.atomShell = new shell[noShl];
  myAtom.shellNumber = new double[noFnc];
  myAtom.shellFunction = new double[noFnc];
  myAtom.coef = new double[noFnc];
  myAtom.exp = new double[noFnc];

  cout << "Atom initialization successfull.\n";
}

void fillAtom(atom myAtom,
              int angs[], double xs[], double ys[], double zs[],
              double exps[], double coefs[]){
  for(int i=0; i<myAtom.noShl; i++){
    myAtom.atomShell[i].ang = angs[i];
    myAtom.atomShell[i].x = xs[i];
    myAtom.atomShell[i].y = ys[i];
    myAtom.atomShell[i].z = zs[i];
  }

  for(int i=0; i<myAtom.noFnc; i++){
    myAtom.exp[i] = exps[i];
    myAtom.coef[i] = coefs[i];
  }

  cout << "Atom fed with data.\n";
}

/// for debugging, checking data initialization
void printAtom(atom myAtom){
  cout << "shells [no, ang, x, y, z]:\n";
  for(int i=0; i<myAtom.noShl; i++){
    cout << myAtom.atomShell[i].ang << "  ";
    cout << myAtom.atomShell[i].x << "  ";
    cout << myAtom.atomShell[i].y << "  ";
    cout << myAtom.atomShell[i].z << "\n";
  }

  cout << "contraction functions [shell no, fnc no, exp, coef]:\n";
  for(int i=0; i<myAtom.noFnc; i++){
    cout << myAtom.shellNumber[i] << "  " << myAtom.shellFunction[i] << "  ";
    cout << myAtom.exp[i] << "  " << myAtom.coef[i] << "\n";
  }

  cout << "===============================";
}
