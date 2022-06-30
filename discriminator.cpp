#include "discriminator.h"

Ptr3D convolution(Ptr3D X, Ptr4D F, int padding, int stride) {
	int outSize = ((X.width - F.width + 2 * padding) / stride) + 1;
	Ptr3D tmp(F.depth2, outSize, outSize);
	int I = tmp.depth;
	int J = tmp.height;
	int K = tmp.width;
	int L = X.depth;
	int M = F.height;
	int N = F.width;

	Ptr3D padX(X.depth, X.height + (padding * 2), X.width + (padding * 2));
	for (int i = 0; i < X.depth; i++) {
		for (int j = 0; j < X.height; j++) {
			for (int k = 0; k < X.width; k++) {
				padX[i][j + padding][k + padding] = X[i][j][k];
			}
		}
	}
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				for (int l = 0; l < L; l++) {
					for (int m = 0; m < M; m++) {
						for (int n = 0; n < N; n++) {
							tmp[i][j][k] += padX[l][j * stride + m][k * stride + n] * F[i][l][m][n];
						}
					}
				}
			}
		}
	}

	return tmp;
}

Ptr2D calError_2D(Ptr2D E_in, Ptr2D X)
{
	Ptr2D tmp(E_in.height, E_in.width);
	for (int i = 0; i < E_in.height; i++) {
		for (int j = 0; j < E_in.width; j++) {
			tmp[i][j] = E_in[i][j] * Leakydrelu2D(X)[i][j];
		}
	}
	return tmp;
}

Ptr2D calWeightDiff(Ptr2D E, Ptr2D A)
{
	Ptr2D tmp(A.width, E.width);
	double learninglate = 0.001;
	for (int i = 0; i < tmp.height; i++) {
		for (int j = 0; j < tmp.width; j++) {
			tmp[i][j] = learninglate * E[0][j] * A[0][i];
		}
	}
	return tmp;
}

Ptr2D calError_in_2D(Ptr2D E, Ptr2D W)
{
	Ptr2D tmp(E.height, W.height);
	for (int i = 0; i < W.height; i++) {
		for (int j = 0; j < W.width; j++) {
			tmp[0][i] += E[0][j] * W[i][j];
		}
	}
	return tmp;
}

void updateWeight(Ptr2D &W, Ptr2D &dW)
{
	for (int i = 0; i < W.height; i++) {
		for (int j = 0; j < W.width; j++) {
			W[i][j] -= dW[i][j];
			dW[i][j] = 0;
		}
	}
}

Ptr3D calError(Ptr3D E_in, Ptr3D X)
{
	Ptr3D tmp(X.depth, X.height, X.width);
	Ptr3D tmp2 = Leakydrelu(X);
	for (int i = 0; i < tmp.depth; i++) {
		for (int j = 0; j < tmp.height; j++) {
			for (int k = 0; k < tmp.width; k++) {
				tmp[i][j][k] = E_in[i][j][k] * tmp2[i][j][k];
			}
		}
	}
	return tmp;
}

Ptr4D calFilterDiff(Ptr4D F, Ptr3D E, Ptr3D X, int stride, int pad)
{
	Ptr4D tmp(F.depth2, F.depth, F.height, F.width);
	double learningrate = 0.001;
	Ptr3D padX(X.depth, X.height + (pad * 2), X.width + (pad * 2));
	for (int i = 0; i < X.depth; i++) {
		for (int j = 0; j < X.height; j++) {
			for (int k = 0; k < X.width; k++) {
				padX[i][j + pad][k + pad] = X[i][j][k];
			}
		}
	}
	for (int i = 0; i < tmp.depth2; i++) {
		for (int j = 0; j < tmp.depth; j++) {
			for (int k = 0; k < tmp.height; k++) {
				for (int l = 0; l < tmp.width; l++) {
					for (int m = 0; m < E.height; m++) {
						for (int n = 0; n < E.width; n++) {
							tmp[i][j][k][l] += learningrate * E[i][m][n] * padX[j][k * stride + m][l * stride + n];
						}
					}
				}
			}
		}
	}
	return tmp;
}

Ptr3D calError_in(Ptr3D E, Ptr4D & F, int stride, int pad)
{
	int outSize = ((E.width - F.width + 2 * pad) / stride) + 1;
	Ptr3D tmp(F.depth, outSize, outSize);
	Ptr3D padX(E.depth, E.height + (pad * 2), E.width + (pad * 2));
	for (int i = 0; i < E.depth; i++) {
		for (int j = 0; j < E.height; j++) {
			for (int k = 0; k < E.width; k++) {
				padX[i][j + pad][k + pad] = E[i][j][k];
			}
		}
	}
	for (int i = 0; i < tmp.depth; i++) {
		for (int j = 0; j < tmp.height; j++) {
			for (int k = 0; k < tmp.width; k++) {
				for (int l = 0; l < F.depth2; l++) {
					for (int m = 0; m < F.height; m++) {
						for (int n = 0; n < F.width; n++) {
							tmp[i][j][k] += padX[l][j * stride + m][k * stride + n] * F[l][i][F.height - m - 1][F.width - n - 1];
						}
					}
				}
			}
		}
	}
	return tmp;
}

void updateFilter(Ptr4D &F, Ptr4D &dF)
{
	for (int i = 0; i < F.depth2; i++) {
		for (int j = 0; j < F.depth; j++) {
			for (int k = 0; k < F.height; k++) {
				for (int l = 0; l < F.width; l++) {
					F[i][j][k][l] -= dF[i][j][k][l];
					dF[i][j][k][l] = 0;
				}
			}
		}
	}
}

Ptr3D Leakyrelu(Ptr3D x)
{
	Ptr3D tmp(x.depth, x.height, x.width);
	for (int i = 0; i < x.depth; i++) {
		for (int j = 0; j < x.height; j++) {
			for (int k = 0; k < x.width; k++) {
				if (x[i][j][k] > 0)tmp[i][j][k] = x[i][j][k];
				else tmp[i][j][k] = 0.2 * tmp[i][j][k];
			}
		}
	}
	return tmp;
}

Ptr2D Leakyrelu2D(Ptr2D x)
{
	Ptr2D tmp(x.height, x.width);
	for (int i = 0; i < x.height; i++) {
		for (int j = 0; j < x.width; j++) {
			if (x[i][j] > 0)tmp[i][j] = x[i][j];
			else tmp[i][j] = 0.2 * tmp[i][j];
		}
	}
	return tmp;
}



Ptr2D softmax(Ptr2D x)
{
	Ptr2D tmp(1, x.width);
	double sum = 0;
	for (int i = 0; i < x.width; i++) {
		double n = x[0][i];
		sum += exp(n);
	}
	for (int i = 0; i < x.width; i++) {
		double n = x[0][i];
		tmp[0][i] = exp(n) / sum;
	}
	return tmp;
}

Ptr2D Leakydrelu2D(Ptr2D x)
{
	Ptr2D tmp(x.height, x.width);
	for (int i = 0; i < x.height; i++) {
		for (int j = 0; j < x.width; j++) {
			if (x[i][j] > 0) tmp[i][j] = 1;
			else tmp[i][j] = 0.2;
		}
	}
	return tmp;
}

Ptr3D Leakydrelu(Ptr3D x)
{
	Ptr3D tmp(x.depth, x.height, x.width);
	for (int i = 0; i < x.depth; i++) {
		for (int j = 0; j < x.height; j++) {
			for (int k = 0; k < x.width; k++) {
				if (x[i][j][k] > 0) tmp[i][j][k] = 1;
				else tmp[i][j][k] = 0.2;
			}
		}
	}
	return tmp;
}

double calcRMSE(Ptr2D L)
{
	int Nrow = L.height;
	int Ncol = L.width;

	int TotNum = Nrow * Ncol;

	double mse = 0;
	for (int i = 0; i < Nrow; i++) {
		for (int j = 0; j < Ncol; j++) {
			mse += (L[i][j] * L[i][j]);
		}
	}

	double rmse = mse / TotNum;
	return rmse;
}

Ptr3D bNormalize(Ptr3D x)
{
	Ptr3D tmp(x.depth, x.height, x.width); Ptr3D tmp(x.depth, x.height, x.width);
	int I = x.depth;
	int J = x.height;
	int K = x.width;
	double sum = 0;

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				sum += x[i][j][k];
			}
		}
	}

	double mean = sum / I*J*K;
	double var = 0;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				var += (x[i][j][k] - mean)*(x[i][j][k] - mean) / I*J*K;
			}
		}
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				tmp[i][j][k] = (x[i][j][k] - mean) / (sqrt(var + 1e-8));
			}
		}
	}

	return tmp;
}


