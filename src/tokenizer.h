#if !defined(TOKENIZER_H)
#define TOKENIZER_H

#include "allocator.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	void tokenizer_encode(ExecCtx* e_ctx, const char* input);
	void tokenizer_decode(ExecCtx* e_ctx, const char* input);

#if defined(__cplusplus)
}
#endif

#endif
