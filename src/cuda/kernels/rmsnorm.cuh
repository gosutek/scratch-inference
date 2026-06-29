#pragma once

#include <cuda_bf16.h>

#include "helpers.h"

typedef __nv_bfloat16 bf16;

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

__global__ void k_rmsnorm(
	bf16* const __restrict__ input_embeddings,
	const u32 dim,
	const bf16* const __restrict__ norm_weights);
