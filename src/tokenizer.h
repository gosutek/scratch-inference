#if !defined(TOKENIZER_H)
#define TOKENIZER_H

#include "allocator.h"
#include "model.cuh"

#if defined(__cplusplus)
extern "C"
{
#endif

	void tokenizer_encoder(ExecCtx* e_ctx, const char* input);
	void tokenizer_decoder(ExecCtx* e_ctx, const char* input);

#if defined(__cplusplus)
}
#endif

#endif
