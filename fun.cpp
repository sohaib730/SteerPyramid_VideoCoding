#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
 #include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include<stdlib.h>


using namespace cv;
using namespace std;
#define pi 3.1416

#define ii sqrt(-1) //for i^nbands in orientation calculation

float rela[10] = {-1,0,1,0,-1,0,1,0,-1,0};   // i^(0 to 9) real part
float imagn[10] = {0,-1,0,1,0,-1,0,-1,0,-1};// i^(0 to 9) imaginary part

float Xrcos[259] = { -1.003906, -1.000000, -0.996094, -0.992188, -0.988281, -0.984375, -0.980469, -0.976563, -0.972656, -0.968750, -0.964844, -0.960938, -0.957031, -0.953125, -0.949219, -0.945313, -0.941406, -0.937500, -0.933594, -0.929688, -0.925781, -0.921875, -0.917969, -0.914063, -0.910156, -0.906250, -0.902344, -0.898438, -0.894531, -0.890625, -0.886719, -0.882813, -0.878906, -0.875000, -0.871094, -0.867188, -0.863281, -0.859375, -0.855469, -0.851563,
-0.847656, -0.843750, -0.839844, -0.835938, -0.832031, -0.828125, -0.824219, -0.820313, -0.816406, -0.812500, -0.808594, -0.804688, -0.800781, -0.796875, -0.792969, -0.789063, -0.785156, -0.781250, -0.777344, -0.773438, -0.769531, -0.765625, -0.761719, -0.757813, -0.753906, -0.750000, -0.746094, -0.742188, -0.738281, -0.734375, -0.730469, -0.726563, -0.722656, -0.718750, -0.714844, -0.710938, -0.707031, -0.703125, -0.699219, -0.695313,
-0.691406, -0.687500, -0.683594, -0.679688, -0.675781, -0.671875, -0.667969, -0.664063, -0.660156, -0.656250, -0.652344, -0.648438, -0.644531, -0.640625, -0.636719, -0.632813, -0.628906, -0.625000, -0.621094, -0.617188, -0.613281, -0.609375, -0.605469, -0.601563, -0.597656, -0.593750, -0.589844, -0.585938, -0.582031, -0.578125, -0.574219, -0.570313, -0.566406, -0.562500, -0.558594, -0.554688, -0.550781, -0.546875, -0.542969, -0.539063,
-0.535156, -0.531250, -0.527344, -0.523438, -0.519531, -0.515625, -0.511719, -0.507813, -0.503906, -0.500000, -0.496094, -0.492188, -0.488281, -0.484375, -0.480469, -0.476563, -0.472656, -0.468750, -0.464844, -0.460938, -0.457031, -0.453125, -0.449219, -0.445313, -0.441406, -0.437500, -0.433594, -0.429688, -0.425781, -0.421875, -0.417969, -0.414063, -0.410156, -0.406250, -0.402344, -0.398438, -0.394531, -0.390625, -0.386719, -0.382813,
-0.378906, -0.375000, -0.371094, -0.367188, -0.363281, -0.359375, -0.355469, -0.351563, -0.347656, -0.343750, -0.339844, -0.335938, -0.332031, -0.328125, -0.324219, -0.320313, -0.316406, -0.312500, -0.308594, -0.304688, -0.300781, -0.296875, -0.292969, -0.289063, -0.285156, -0.281250, -0.277344, -0.273438, -0.269531, -0.265625, -0.261719, -0.257813, -0.253906, -0.250000, -0.246094, -0.242188, -0.238281, -0.234375, -0.230469, -0.226563,
-0.222656, -0.218750, -0.214844, -0.210938, -0.207031, -0.203125, -0.199219, -0.195313, -0.191406, -0.187500, -0.183594, -0.179687, -0.175781, -0.171875, -0.167969, -0.164063, -0.160156, -0.156250, -0.152344, -0.148438, -0.144531, -0.140625, -0.136719, -0.132812, -0.128906, -0.125000, -0.121094, -0.117188, -0.113281, -0.109375, -0.105469, -0.101563, -0.097656, -0.093750, -0.089844, -0.085938, -0.082031, -0.078125, -0.074219, -0.070312,
-0.066406, -0.062500, -0.058594, -0.054688, -0.050781, -0.046875, -0.042969, -0.039063, -0.035156, -0.031250, -0.027344, -0.023438, -0.019531, -0.015625, -0.011719, -0.007812, -0.003906, 0.000000, 0.003906 };
//for Highpass
float Yrcos[259] = { 0.000000, 0.000000, 0.006136, 0.012272, 0.018407, 0.024541, 0.030675, 0.036807, 0.042938, 0.049068, 0.055195, 0.061321, 0.067444, 0.073565, 0.079682, 0.085797, 0.091909, 0.098017, 0.104122, 0.110222, 0.116319, 0.122411, 0.128498, 0.134581, 0.140658, 0.146730, 0.152797, 0.158858, 0.164913, 0.170962, 0.177004, 0.183040, 0.189069, 0.195090, 0.201105, 0.207111, 0.213110, 0.219101, 0.225084, 0.231058,
0.237024, 0.242980, 0.248928, 0.254866, 0.260794, 0.266713, 0.272621, 0.278520, 0.284408, 0.290285, 0.296151, 0.302006, 0.307850, 0.313682, 0.319502, 0.325310, 0.331106, 0.336890, 0.342661, 0.348419, 0.354164, 0.359895, 0.365613, 0.371317, 0.377007, 0.382683, 0.388345, 0.393992, 0.399624, 0.405241, 0.410843, 0.416430, 0.422000, 0.427555, 0.433094, 0.438616, 0.444122, 0.449611, 0.455084, 0.460539,
0.465976, 0.471397, 0.476799, 0.482184, 0.487550, 0.492898, 0.498228, 0.503538, 0.508830, 0.514103, 0.519356, 0.524590, 0.529804, 0.534998, 0.540171, 0.545325, 0.550458, 0.555570, 0.560662, 0.565732, 0.570781, 0.575808, 0.580814, 0.585798, 0.590760, 0.595699, 0.600616, 0.605511, 0.610383, 0.615232, 0.620057, 0.624859, 0.629638, 0.634393, 0.639124, 0.643832, 0.648514, 0.653173, 0.657807, 0.662416,
0.667000, 0.671559, 0.676093, 0.680601, 0.685084, 0.689541, 0.693971, 0.698376, 0.702755, 0.707107, 0.711432, 0.715731, 0.720003, 0.724247, 0.728464, 0.732654, 0.736817, 0.740951, 0.745058, 0.749136, 0.753187, 0.757209, 0.761202, 0.765167, 0.769103, 0.773010, 0.776888, 0.780737, 0.784557, 0.788346, 0.792107, 0.795837, 0.799537, 0.803208, 0.806848, 0.810457, 0.814036, 0.817585, 0.821103, 0.824589,
0.828045, 0.831470, 0.834863, 0.838225, 0.841555, 0.844854, 0.848120, 0.851355, 0.854558, 0.857729, 0.860867, 0.863973, 0.867046, 0.870087, 0.873095, 0.876070, 0.879012, 0.881921, 0.884797, 0.887640, 0.890449, 0.893224, 0.895966, 0.898674, 0.901349, 0.903989, 0.906596, 0.909168, 0.911706, 0.914210, 0.916679, 0.919114, 0.921514, 0.923880, 0.926210, 0.928506, 0.930767, 0.932993, 0.935184, 0.937339,
0.939459, 0.941544, 0.943593, 0.945607, 0.947586, 0.949528, 0.951435, 0.953306, 0.955141, 0.956940, 0.958703, 0.960431, 0.962121, 0.963776, 0.965394, 0.966976, 0.968522, 0.970031, 0.971504, 0.972940, 0.974339, 0.975702, 0.977028, 0.978317, 0.979570, 0.980785, 0.981964, 0.983105, 0.984210, 0.985278, 0.986308, 0.987301, 0.988258, 0.989177, 0.990058, 0.990903, 0.991710, 0.992480, 0.993212, 0.993907,
0.994565, 0.995185, 0.995767, 0.996313, 0.996820, 0.997290, 0.997723, 0.998118, 0.998476, 0.998795, 0.999078, 0.999322, 0.999529, 0.999699, 0.999831, 0.999925, 0.999981, 1.000000, 1.000000 };
//for lowpass
float YIrcos[259] = { 1.000000, 1.000000, 0.999981, 0.999925, 0.999831, 0.999699, 0.999529, 0.999322, 0.999078, 0.998795, 0.998476, 0.998118, 0.997723, 0.997290, 0.996820, 0.996313, 0.995767, 0.995185, 0.994565, 0.993907, 0.993212, 0.992480, 0.991710, 0.990903, 0.990058, 0.989177, 0.988258, 0.987301, 0.986308, 0.985278, 0.984210, 0.983105, 0.981964, 0.980785, 0.979570, 0.978317, 0.977028, 0.975702, 0.974339, 0.972940,
0.971504, 0.970031, 0.968522, 0.966976, 0.965394, 0.963776, 0.962121, 0.960431, 0.958703, 0.956940, 0.955141, 0.953306, 0.951435, 0.949528, 0.947586, 0.945607, 0.943593, 0.941544, 0.939459, 0.937339, 0.935184, 0.932993, 0.930767, 0.928506, 0.926210, 0.923880, 0.921514, 0.919114, 0.916679, 0.914210, 0.911706, 0.909168, 0.906596, 0.903989, 0.901349, 0.898674, 0.895966, 0.893224, 0.890449, 0.887640,
0.884797, 0.881921, 0.879012, 0.876070, 0.873095, 0.870087, 0.867046, 0.863973, 0.860867, 0.857729, 0.854558, 0.851355, 0.848120, 0.844854, 0.841555, 0.838225, 0.834863, 0.831470, 0.828045, 0.824589, 0.821103, 0.817585, 0.814036, 0.810457, 0.806848, 0.803208, 0.799537, 0.795837, 0.792107, 0.788346, 0.784557, 0.780737, 0.776888, 0.773010, 0.769103, 0.765167, 0.761202, 0.757209, 0.753187, 0.749136,
0.745058, 0.740951, 0.736817, 0.732654, 0.728464, 0.724247, 0.720003, 0.715731, 0.711432, 0.707107, 0.702755, 0.698376, 0.693971, 0.689541, 0.685084, 0.680601, 0.676093, 0.671559, 0.667000, 0.662416, 0.657807, 0.653173, 0.648514, 0.643832, 0.639124, 0.634393, 0.629638, 0.624859, 0.620057, 0.615232, 0.610383, 0.605511, 0.600616, 0.595699, 0.590760, 0.585798, 0.580814, 0.575808, 0.570781, 0.565732,
0.560662, 0.555570, 0.550458, 0.545325, 0.540171, 0.534998, 0.529804, 0.524590, 0.519356, 0.514103, 0.508830, 0.503538, 0.498228, 0.492898, 0.487550, 0.482184, 0.476799, 0.471397, 0.465976, 0.460539, 0.455084, 0.449611, 0.444122, 0.438616, 0.433094, 0.427555, 0.422000, 0.416430, 0.410843, 0.405241, 0.399624, 0.393992, 0.388345, 0.382683, 0.377007, 0.371317, 0.365613, 0.359895, 0.354164, 0.348419,
0.342661, 0.336890, 0.331106, 0.325310, 0.319502, 0.313682, 0.307850, 0.302006, 0.296151, 0.290285, 0.284408, 0.278520, 0.272621, 0.266713, 0.260794, 0.254866, 0.248928, 0.242980, 0.237024, 0.231058, 0.225084, 0.219101, 0.213110, 0.207111, 0.201105, 0.195090, 0.189069, 0.183040, 0.177004, 0.170962, 0.164913, 0.158858, 0.152797, 0.146730, 0.140658, 0.134581, 0.128498, 0.122411, 0.116319, 0.110222,
0.104122, 0.098017, 0.091909, 0.085797, 0.079682, 0.073565, 0.067444, 0.061321, 0.055195, 0.049068, 0.042938, 0.036807, 0.030675, 0.024541, 0.018407, 0.012272, 0.006136, 0.000000, 0.000000};
//factorial lookup table
//factorial lookup table
float facto[13] = { 1.000000, 1.000000, 2.000000, 6.000000, 24.000000,
120.000000, 720.000000, 5040.000000, 40320.000000, 362880.000000,
3628800.000000, 39916800.000000, 479001600.000000 };
float Ycosn[3075];
float YIrcos_abs[259];


Mat pointop(Mat fa, float lut[], float origin, float increment,int nsize)
{
	//float *res= new float [nsize];

	int lutsize = 257, index;
	float pos;
	Mat res(fa.rows,fa.cols,CV_32F);
	if (increment > 0)
	{
		for (int i = 0; i < nsize; i++)
		{
			pos = (fa.at<float>(i) - origin) / increment;
			index = (int)pos;
			if (index < 0)
			{
				index = 0;
			}
			else if (index > lutsize)
			{
				index = lutsize;
			}
			
			
				res.at<float>(i) =lut[index]+ (lut[index + 1] - lut[index]) * (pos - index);
			
		}
	}
	else
	{
		for (int i = 0; i < nsize; i++)
		{
			res.at<float>(i) =(float) lut[0];
			
		}
	}

	
	return res;
	
}


Mat fftshift(Mat input)
{
int m,n,xMid,yMid;
m=input.cols;
n=input.rows;
xMid=m>>1;
yMid=n>>1;
 Mat tmp;
            Mat q0(input, Rect(0,    0,    xMid, yMid));
            Mat q1(input, Rect(xMid, 0,    xMid, yMid));
            Mat q2(input, Rect(0,    yMid, xMid, yMid));
            Mat q3(input, Rect(xMid, yMid, xMid, yMid));

           q0.copyTo(tmp);
           q3.copyTo(q0);
            tmp.copyTo(q3);

          q1.copyTo(tmp);                    
          q2.copyTo(q1);
          tmp.copyTo(q2);

			q0.copyTo(input(Rect(0,    0,    xMid, yMid)));
			q1.copyTo(input(Rect(xMid, 0,    xMid, yMid)));
			q2.copyTo(input(Rect(0,    yMid, xMid, yMid)));
			q3.copyTo(input(Rect(xMid, yMid, xMid, yMid)));

			return input;


}

Mat ifftshift(Mat input)
{
int m,n,xMid,yMid;
m=input.cols;
n=input.rows;
xMid=m>>1;
yMid=n>>1;
 Mat tmp;
            Mat q0(input, Rect(0,    0,    xMid, yMid));
            Mat q1(input, Rect(xMid, 0,    xMid, yMid));
            Mat q2(input, Rect(0,    yMid, xMid, yMid));
            Mat q3(input, Rect(xMid, yMid, xMid, yMid));

            q3.copyTo(tmp);
            q0.copyTo(q3);
            tmp.copyTo(q0);

            q2.copyTo(tmp);
            q1.copyTo(q2);
            tmp.copyTo(q1);


			q0.copyTo(input(Rect(0,    0,    xMid, yMid)));
			q1.copyTo(input(Rect(xMid, 0,    xMid, yMid)));
			q2.copyTo(input(Rect(0,    yMid, xMid, yMid)));
			q3.copyTo(input(Rect(xMid, yMid, xMid, yMid)));

			return input;


}

Mat buildSCFpyrlevs(Mat ima,Mat *lodft, Mat radd, float *Xrcos, float *Yrcos, Mat ang, int ht, int nbands,int m,int n)
{


	
	int nsize=m*n;
	Mat manc;
	merge(lodft,2,manc);

	/*
		for (int i = 0; i < 258; i++)
		{
			Xrcos[i] = Xrcos[i] -1;

		}

		int order = nbands - 1;
		float consta, alp;
		consta = pow(2.0, order * 2)*pow(facto[order], 2) / (nbands*facto[2 * order]);
		
		//Used for orientation interpolation
		for (int i = 0; i < 3074; i++)
		{
			alp = fabs(fmod(pi + Xcosn[i], 2 * pi)) - pi;
			Ycosn[i] = 2 * sqrt(consta)*pow(cos(Xcosn[i]), order)*(fabs(alp) < pi / 2);
			
		}
		Mat lodftx[2];
		lodft[0].copyTo(lodftx[0]);
		lodft[1].copyTo(lodftx[1]);
		Mat himask1(nsize,1,CV_32F);
		Mat lomask1(nsize,1,CV_32F);
		Mat rad2d,ang2d;
		himask1=pointop(radd,YIrcos,Xrcos[0],Xrcos[1]-Xrcos[0],nsize);
		Mat lomask(m,n,CV_32F);
		Mat himask(m,n,CV_32F);
		Mat anglemask(m,n,CV_32F);
		himask=himask1.reshape(0,m);
		Mat anglemask1(nsize,1,CV_32F);
		Mat tmp(m,n,CV_32F);
		Mat banddft[2];
		Mat band;

		Mat interx;
		for(int i=0;i<nbands;i++)
		{
			anglemask1=pointop(radd,Ycosn, Xcosn[0]+(pi*i/nbands),Xrcos[1]-Xrcos[0],nsize);
			anglemask=anglemask1.reshape(0,m);
			tmp=anglemask.mul(himask);
			banddft[0]=lodftx[0]*rela[nbands]-lodftx[1]*imagn[nbands];
			banddft[0]=banddft[0].mul(tmp);
			banddft[1]=lodftx[1]*rela[nbands]+lodftx[0]*imagn[nbands];
			banddft[0]=banddft[0].mul(tmp);
			banddft[0]=ifftshift(banddft[0]);
			banddft[1]=ifftshift(banddft[1]);
			merge(banddft,2,interx);
			dft(interx, band, DFT_INVERSE | DFT_REAL_OUTPUT ); // Applying IDFT
			
		}//for nbands
	//saving part

	//scaling part

	*/
Mat rad2d=radd.reshape(0,m);
Mat ang2d=ang.reshape(0,m);

int sa=ceil(m/2.0),sb=ceil(n/2.0);
int cst=ceil(sb/2.0),rst=ceil(sa/2.0);


m=sa;
n=sb;
nsize=m*n;
	Mat dummy_lodftr(lodft[0], Rect(cst,rst,n, m));
	Mat dummy_lodftim(lodft[1], Rect(cst,rst,n, m));
	Mat dummy_angl(ang2d, Rect(cst,rst,n, m));
	Mat dummy_radd(rad2d, Rect(cst,rst,n, m));

	Mat ra2,an2;

	dummy_angl.copyTo(an2);
	dummy_radd.copyTo(ra2);

	Mat ra1=ra2.reshape(0,1);
	Mat an1=an2.reshape(0,1);
	
	
	
	
	Mat lodft1[]={Mat_<float>(dummy_angl), Mat::zeros(dummy_angl.size(), CV_32F)};

	dummy_lodftr.copyTo(lodft[0]);
	dummy_lodftim.copyTo(lodft1[1]);
	for (int i = 0; i < 259;i++)
		{
			YIrcos_abs[i] = fabs(YIrcos[i]);
		}

	Mat lomask1=pointop(ra1,YIrcos_abs,Xrcos[0],Xrcos[1]-Xrcos[0],nsize);

	Mat lomask=lomask1.reshape(0,m);

	

	lodft1[0]=lodft1[0].mul(lomask);
	lodft1[1]=lodft1[1].mul(lomask);

	lodft1[0]=ifftshift(lodft1[0]);
	lodft1[1]=ifftshift(lodft1[1]);
	Mat xda,inverseDFT;
	merge(lodft1,2,xda);


	dft(xda, inverseDFT, DFT_INVERSE | DFT_REAL_OUTPUT ); // Applying IDFT
	
   
	
	Mat xx(m*2,n*2,CV_8U);

	inverseDFT.convertTo(xx,CV_8U,255.0,0);
	imwrite("lo.png", xx);

	

	

 	




	

	

return ima;
}



Mat buildpyr(Mat im)
{
	int m=im.rows;
	int n=im.cols;
	int ht=3,nbands=4;
	int nsize=m*n;
	Mat imf(m,n,CV_32F),img(m,n,CV_8U);
	im.convertTo(img,CV_8U);
	img.convertTo(imf, CV_32F,1.0/255,0);
    Mat ima(m,n,CV_32F);

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//Compute radius and angle matrix base for a given size of m,n
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	int centm=ceil(m/2.0);
	int centn=ceil(n/2.0);
	float *r = new float[n];
	float *t = new float[n];
	float *r1 = new float[m*n];
	float *t1 = new float[m*n];
	float *rad = new float[m*n]; //polar cooridinates row component
	float *angl = new float[m*n];// polar coordinates angle component
	


	for (int i = -n / 2; i < n / 2; i++)
	{
		r[i + n / 2] = (float)(i + 1) / n;
		r[i + n / 2] *= 2;
		for (int j = 0; j < m; j++)
		{
			r1[(i + n / 2) + j*n] = r[i + n / 2];


		}

	}

	for (int i = -m / 2; i < m / 2; i++)
	{
		t[i + m / 2] = (float)(i + 1) / m;
		t[i + m / 2] *= 2;
		for (int j = 0; j < n; j++)
		{
			t1[(i + m / 2)*n + j] = t[i + m / 2];

		}
	}


	
	for (int i = 0; i < m*n; i++)
	{
		float temp = sqrt(r1[i] * r1[i] + t1[i] * t1[i]);
		if (temp != 0)
		{
			rad[i] = log10(temp)/log10(2.0);
		}
		else
		{
			rad[i] = rad[i - 1];
		}
		angl[i] = atan2(t1[i], r1[i]);

	}

	//Convert 1D to 2D
	Mat angld(nsize,1,CV_32F),radd(nsize,1,CV_32F),lo1d(nsize,1,CV_32F),hi1d(nsize,1,CV_32F);
	

	for(int i=0;i<nsize;i++)
	{
		angld.at<float>(i,0)=angl[i];
		radd.at<float>(i,0)=rad[i];
	}

	lo1d=pointop(radd,YIrcos,Xrcos[0],Xrcos[1]-Xrcos[0],nsize);
	hi1d=pointop(radd,Yrcos,Xrcos[0],Xrcos[1]-Xrcos[0],nsize);
	
	//2d of above lowpass and highass kernals
	Mat lo2d(m,n,CV_32F),hi2d(m,n,CV_32F);

	lo2d=lo1d.reshape(0,im.rows);
	hi2d=hi1d.reshape(0,im.rows);
	
	
	
	//DFT of the image 
	Mat dft_im,inverseDFT;
	Mat re_ima_dft[] = {Mat_<float>(im), Mat::zeros(im.size(), CV_32F)};
	Mat lowpas_1[] = {Mat_<float>(im), Mat::zeros(im.size(), CV_32F)};
	Mat hipass_1[] = {Mat_<float>(im), Mat::zeros(im.size(), CV_32F)};
	
	dft(imf, dft_im,DFT_SCALE| DFT_COMPLEX_OUTPUT); 
	split(dft_im,re_ima_dft);   //split fft to real and imaginary parts
	double min, max;

	
   re_ima_dft[0]=fftshift(re_ima_dft[0]);
   re_ima_dft[1]=fftshift(re_ima_dft[1]);

   	
	lowpas_1[0]=re_ima_dft[0].mul(lo2d);
	lowpas_1[1]=re_ima_dft[1].mul(lo2d);
	
	hipass_1[0]=re_ima_dft[0].mul(hi2d);
	hipass_1[1]=re_ima_dft[1].mul(hi2d);


		

	ima=buildSCFpyrlevs(ima,lowpas_1,radd,Xrcos,Yrcos,angld,ht-1,nbands,m,n);

	/* lowpas_1[0]=ifftshift(lowpas_1[0]);
	lowpas_1[1]=ifftshift(lowpas_1[1]);

	Mat ddd;
	merge(lowpas_1,2,ddd);

	dft(ddd, inverseDFT, DFT_INVERSE | DFT_REAL_OUTPUT ); // Applying IDFT
	
   
	
	Mat aa(m,n,CV_8U);
	inverseDFT.convertTo(aa,CV_8U,255,0);
	

 	namedWindow( "Gray image", CV_WINDOW_AUTOSIZE );
	imshow( "Gray image", aa );*/


	
	return ima;
	


}