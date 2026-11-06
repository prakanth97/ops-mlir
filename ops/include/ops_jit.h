/** @file
  * @brief Public C API of libopsjit, the MLIR JIT chain executor for OPS.
  * @details Applications translated with `ops-translator --jit` reference ops_jit_init() from the generated
  * ops_init_backend(), so linking libopsjit is required for them. Everything else is configured through
  * environment variables (OPS_JIT_*, see docs/user-guide.md) or ops_jit_set_option().
  */
#ifndef OPS_JIT_H
#define OPS_JIT_H

#ifdef __cplusplus
extern "C" {
#endif

/** Reads the options and registers the chain executor with OPS. OPS_JIT=0 registers nothing. Idempotent. */
void ops_jit_init(void);

/** Sets an option by the name of its environment variable (e.g. "OPS_JIT_FUSION", "0"). Returns 0 on success, 1 for
 *  an unknown name or an invalid value. Takes effect for loops queued afterwards. */
int ops_jit_set_option(const char *name, const char *value);

/** Prints the counters and phase times (also done at exit with OPS_JIT_STATS=1). */
void ops_jit_print_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* OPS_JIT_H */
