#pragma once

#include "cuda_helpers.cuh"

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

__global__ void k_fetch_input_embeddings_v1(
	const u32* const __restrict__ input_tokens,
	const u64 input_tokens_len,
	const u32 dim,
	const bf16* const __restrict__ embeddings_table,
	bf16* const __restrict__ input_embeddings,
	const u32 stride);

__global__ void k_fetch_input_embeddings(
	const u32* const __restrict__ input_tokens,
	const u64 input_tokens_len,
	const u32 dim,
	const bf16* const __restrict__ embeddings_table,
	bf16* const __restrict__ input_embeddings);
