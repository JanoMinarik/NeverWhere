#ifndef __atom_H_INCLUDED__
#define __atom_H_INCLUDED__

struct shell {
    int ang;
    double x;
    double y;
    double z;
};

struct atom {
    int noShl;
    int noFnc;
    /// data
    shell *atomShell;
    int *shellNumber;
    int *shellFunction;
    double *coef;
    double *exp;
};

void printAtom(atom);

#endif
