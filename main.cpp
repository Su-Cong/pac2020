/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
//#include <complex>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <xmmintrin.h>

using namespace std;

//typedef complex<double> Complex;
typedef chrono::high_resolution_clock Clock;

const int m=1638400;	// DO NOT CHANGE!!
const int K=100000;	// DO NOT CHANGE!!

double logDataVSPrior(const double* dat_r, const double* dat_i, const double* pri_r, const double* pri_i, const double* ctf, const double* sigRcp, const int num, const double disturb0);

int main ( int argc, char *argv[] )
{ 
//    Complex *dat = new Complex[m];
//    Complex *pri = new Complex[m];
    double *dat_r = new double[m], *dat_i = new double[m];
    double *pri_r = new double[m], *pri_i = new double[m];
    double *ctf = new double[m];
    double *sigRcp = new double[m];
    double *disturb = new double[K];
    double *res = new double[K];
    double dat0, dat1, pri0, pri1, ctf0, sigRcp0;

    /***************************
     * Read data from input.dat
     * *************************/
    ifstream fin;

    fin.open("input.dat");
    if(!fin.is_open())
    {
        cout << "Error opening file input.dat" << endl;
        exit(1);
    }
    int i=0;
    while( !fin.eof() ) 
    {
        fin >> dat0 >> dat1 >> pri0 >> pri1 >> ctf0 >> sigRcp0;
//        dat[i] = Complex (dat0, dat1);
//        pri[i] = Complex (pri0, pri1);
        dat_r[i] = dat0;
	    dat_i[i] = dat1;
		pri_r[i] = pri0;
		pri_i[i] = pri1;
        ctf[i] = ctf0;
        sigRcp[i] = sigRcp0;
        i++;
        if(i == m) break;
    }
    fin.close();

    fin.open("K.dat");
    if(!fin.is_open())
    {
	cout << "Error opening file K.dat" << endl;
	exit(1);
    }
    i=0;
    while( !fin.eof() )
    {
	fin >> disturb[i];
	i++;
	if(i == K) break;
    }
    fin.close();

    /***************************
     * main computation is here
     * ************************/
    auto startTime = Clock::now(); 

    ofstream fout;
    fout.open("result.dat");
    if(!fout.is_open())
    {
         cout << "Error opening file for result" << endl;
         exit(1);
    }

#pragma omp parallel for num_threads(96) schedule(static) 
    for(unsigned int t = 0; t < K; t++)
    {
        res[t] = logDataVSPrior(dat_r, dat_i, pri_r, pri_i, ctf, sigRcp, m, disturb[t]);
    }
    for(unsigned int t = 0; t < K; t++)
    {
        fout << t+1 << ": " << res[t] << endl;
    }
    fout.close();

    auto endTime = Clock::now(); 

    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() << " microseconds" << endl;

    delete[] dat_r;
    delete[] dat_i;
    delete[] pri_r;
    delete[] pri_i;

    delete[] ctf;
    delete[] sigRcp;
    delete[] disturb;
    return EXIT_SUCCESS;
}

double logDataVSPrior(const double* dat_r, const double* dat_i, const double* pri_r, const double* pri_i, const double* ctf, const double* sigRcp, const int num, const double disturb0)
{
    double result = 0.0;

    __m512 dis_v = _mm512_set1_pd(disturb0);
    for(int i = 0; i < num; i+=8)
    {
    	__m512d dat_r_v = _mm512_loadu_pd(dat_r + i);
	    __m512d dat_i_v = _mm512_loadu_pd(dat_i + i);
	    __m512d pri_r_v = _mm512_loadu_pd(pri_r + i);
	    __m512d pri_i_v = _mm512_loadu_pd(pri_i + i);
	    __m512d ctf_v = _mm512_loadu_pd(ctf + i);
	    __m512d sig_v = _mm512_loadu_pd(sigRcp + i);
	    
	    //calculation
	    __m512 mid_v = _mm512_mul_pd(dis_v, ctf_v);
	    __m512 real = _mm512_fnmadd_pd(mid_v, pri_r_v, dat_r_v);
	    __m512 imag = _mm512_fnmadd_pd(mid_v, pri_i_v, dat_i_v);
	    real = _mm512_mul_pd(real, real);
	    iamg = _mm512_mul_pd(imag, imag);
	    real = _mm512_add_pd(real, iamg);
	    result += _mm512_reduce_add_pd(_mm512_mul_pd(real, sig_v));
	}
//#pragma ivdep
//    double r, image;
//    for (int i = 0; i < num; i++)
//    {
//	  r = dat_r[i] - disturb0 * ctf[i] * pri_r[i];
//	  image = dat_i[i] - disturb0 * ctf[i] * pri_i[i];
//	  result += r*r*sigRcp[i] + image*image*sigRcp[i];
//    }
//     for (int i = 0; i < num; i++)
//     {

//           result += ( norm( dat[i] - disturb0 * ctf[i] * pri[i] ) * sigRcp[i] );

//     }
    return result;
}
