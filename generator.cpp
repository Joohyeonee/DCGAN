#include "generator.h"

Ptr3D convTranspose(Ptr3D Y, Ptr4D F, int padding, int stride) {
	int outSize = ((Y.width - F.width + 2 * padding) / stride) + 1;
	Ptr3D MatOut(F.depth2, outSize, outSize);
	int I = MatOut.depth;
	int J = MatOut.height;
	int K = MatOut.width;
	int L = Y.depth;
	int M = F.height;
	int N = F.width;


	/*int O = X.width;
	int I = tmp.depth;
	int J = tmp.height;
	int K = tmp.width;
	int L = X.depth;
	int M = F.height;
	int N = F.width;*/

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				for (int l = 0; l < L; l++) {
					for (int m = 0; m < M; m++) {
						for (int n = 0; n < N; n++) {
							MatOut[i][j][k] += Y[l][j * stride + m][k * stride + n] * F[i][l][m][n];
						}
					}
				}
			}
		}
	}

	return MatOut;
}





