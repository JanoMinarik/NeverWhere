#ifndef __neon_H_INCLUDED__
#define __neon_H_INCLUDED__
struct shell {
    int ang;
    double x;
    double y;
    double z;
};

class Neon
{
public:
    // constructor to initialize Neon data, shell coords are set to 0's.
    Neon();
    // Neon manipulation functions
    void printNeon();
    void printShell();
    void setShell(int, int, double, double, double);
    // raw data
    shell shellCord[6];
    int shellNumber[25];
    int shellFunction[25];
    static double exponent[25];
    static double coefficient[25];
};
#endif
