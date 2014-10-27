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
    Neon();
    void printNeon();
    void setCoord(int, int, double, double, double);

protected:
    shell shellCord[6];
    int shellNumber[25];
    int shellFunction[25];
    static double exponent[25];
    static double coefficient[25];
};

#endif
