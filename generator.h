#pragma once
#include "structs.h"

class generator
{
public:
	Ptr3D convTranspose(Ptr3D Y, Ptr4D F, int padding, int stride);
};

