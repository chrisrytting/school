void cssor(double* U, int m, int n, double omega, double tol, int maxiters, int* info)
{
    double maxerr,temp,lcf,rcf;
    int i,j,k;
    lcf=1.0-omega;
    rcf=0.25*omega;
    for (k=0;k<maxiters;k++)
    {
        maxerr=0.0;
        for (j=1; j<n-1;j++)
        {
            for (i=1;i<m-1;i++)
            {
                temp=U[i*n+j];
                U[i*n+j]=lcf*U[i*n+j]+rcf*(U[i*n+j-1] + U[i*n+j+1]+U[(i-1)*n+j]+U[(i+1)*n+j]);
                maxerr=fmax(fabs(U[i*n+j]-temp),maxerr);
            }
        }

        if (maxerr<tol)
        {
            break;
        }
    }
    if (maxerr<tol)
    {
        *info=0;
    }
    else
    {
        *info=1;
    }


}
