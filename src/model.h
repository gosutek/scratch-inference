#include "helpers.h"

typedef struct Config
{
	u32 n_layers;
} Config;

typedef struct Weights
{
	// Embeddings
	f32* token_embedding_table;
	f32* token_embedding_table_per_layer;

	// RMS
	f32* rms_input;
	f32* rms_q;
	f32* rms_k;
	f32* rms_post_attn;
	f32* rms_pre_ffn;
	f32* rms_post_ffn;

	// Attention
	f32* wq;
	f32* wk;
	f32* wv;
	f32* wo;

	// FFN
	f32* mlp_gate;
	f32* mlp_up;
	f32* mlp_down;

	// TODO: PLE Block
	// ...
	// ...
	// ...

} Weights;

typedef struct Model
{
	Config  config;
	Weights weights;
	i32     fd;
	f32*    data;  // NOTE: Should be bf16? mmap ptr
	u64     file_bsize;
} Model;
