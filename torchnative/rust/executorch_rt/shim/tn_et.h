/*
 * The C surface of torchnative's ExecuTorch runtime shim (issue #315, step R1).
 *
 * Opaque handles and integer status codes only: no C++ type, no exception and
 * no ownership of a Python object crosses this boundary. A status is 0 for
 * success or an `executorch::runtime::Error` value (runtime/core/error.h);
 * the shim never invents a code of its own, so the Rust side maps one table.
 *
 * Lifetimes, which the caller (the Rust wrapper in src/lib.rs) upholds:
 *
 *   caller's program bytes  >=  tn_et_program  >=  every tn_et_method made from it
 *
 * `tn_et_program_load` does not copy: the runtime's BufferDataLoader reads the
 * caller's bytes for as long as the Program lives. A method holds a pointer
 * to its program. Pointers this header hands out (names, sizes, output data)
 * are valid until the handle they came from is freed, and output data only
 * until the next set_input or execute on that method.
 *
 * Threads: a tn_et_method is not reentrant; one caller at a time. A
 * tn_et_program is read-only after load; the Rust wrapper still serializes
 * calls on it. The diagnostic log (`tn_et_take_log`) is per thread.
 */
#ifndef TORCHNATIVE_TN_ET_H
#define TORCHNATIVE_TN_ET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tn_et_program tn_et_program;
typedef struct tn_et_method tn_et_method;

/* executorch::runtime::Tag values (runtime/core/tag.h). */
#define TN_ET_TAG_NONE 0
#define TN_ET_TAG_TENSOR 1
#define TN_ET_TAG_STRING 2
#define TN_ET_TAG_DOUBLE 3
#define TN_ET_TAG_INT 4
#define TN_ET_TAG_BOOL 5

/* Counter families for tn_et_method_counter_*. */
#define TN_ET_COUNT_OPERATOR_CALLS 0 /* per kernel, from "native_call_<op>" scopes */
#define TN_ET_COUNT_SCOPES 1         /* every profiling scope name the runtime opened */
#define TN_ET_COUNT_DELEGATE_OPS 2   /* names a delegate reported (none in R1) */

typedef struct {
  size_t num_inputs;
  size_t num_outputs;
  size_t num_planned_buffers;
  size_t num_backends;
  size_t num_instructions;
} tn_et_method_info;

typedef struct {
  int32_t tag;           /* TN_ET_TAG_* */
  int8_t scalar_type;    /* valid when tag == TN_ET_TAG_TENSOR */
  uint8_t memory_planned;
  size_t ndim;
  const int32_t* sizes;  /* ndim entries, owned by the program */
  const uint8_t* dim_order;
  size_t nbytes;
} tn_et_tensor_info;

typedef struct {
  int32_t tag;
  int8_t scalar_type;
  size_t ndim;
  const int32_t* sizes;
  const uint8_t* dim_order;
  const void* data;
  size_t nbytes;
  int64_t int_value;
  double double_value;
  uint8_t bool_value;
} tn_et_value;

/* Process-wide. Idempotent and thread-safe: installs the shim's PAL hooks
 * (log capture, abort report) and runs runtime_init(). Every other entry
 * point calls it, so a caller never sees an uninitialized PAL. */
int32_t tn_et_init(void);

/* Copies this thread's captured runtime log into buf (NUL-terminated,
 * truncated to cap) and clears it. Returns the untruncated length. */
size_t tn_et_take_log(char* buf, size_t cap);

/* Build facts. */
int32_t tn_et_event_tracer_enabled(void);
int32_t tn_et_program_verification_enabled(void);
size_t tn_et_kernel_count(void);
const char* tn_et_kernel_name(size_t index); /* NULL past the end */
size_t tn_et_backend_count(void);
const char* tn_et_backend_name(size_t index); /* NULL past the end */

/* Program. data must stay alive and unchanged until tn_et_program_free, and
 * must be aligned to alignof(max_align_t) (the runtime refuses otherwise). */
int32_t tn_et_program_load(const uint8_t* data, size_t len, tn_et_program** out);
void tn_et_program_free(tn_et_program* program);
size_t tn_et_program_num_methods(const tn_et_program* program);
const char* tn_et_program_method_name(const tn_et_program* program, size_t index);
int32_t tn_et_method_meta(const tn_et_program* program, const char* name,
                          tn_et_method_info* out);
int32_t tn_et_method_input_meta(const tn_et_program* program, const char* name,
                                size_t index, tn_et_tensor_info* out);
int32_t tn_et_method_output_meta(const tn_et_program* program, const char* name,
                                 size_t index, tn_et_tensor_info* out);
int32_t tn_et_method_planned_buffer_size(const tn_et_program* program,
                                         const char* name, size_t index,
                                         int64_t* out);
int32_t tn_et_method_backend_name(const tn_et_program* program, const char* name,
                                  size_t index, const char** out);

/* Method. The program must outlive the method. */
int32_t tn_et_method_load(const tn_et_program* program, const char* name,
                          tn_et_method** out);
void tn_et_method_free(tn_et_method* method);
size_t tn_et_method_num_inputs(const tn_et_method* method);
size_t tn_et_method_num_outputs(const tn_et_method* method);
/* Copies nbytes from data (one copy, into the planned arena or into a buffer
 * the method owns); data is not referenced after the call returns. */
int32_t tn_et_method_set_input(tn_et_method* method, size_t index,
                               int8_t scalar_type, const int64_t* sizes,
                               size_t ndim, const void* data, size_t nbytes);
int32_t tn_et_method_execute(tn_et_method* method);
int32_t tn_et_method_output(const tn_et_method* method, size_t index,
                            tn_et_value* out);
uint64_t tn_et_method_executions(const tn_et_method* method);
size_t tn_et_method_counter_count(const tn_et_method* method, int32_t family);
int32_t tn_et_method_counter_at(const tn_et_method* method, int32_t family,
                                size_t index, const char** name,
                                uint64_t* count);
void tn_et_method_reset_counters(tn_et_method* method);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TORCHNATIVE_TN_ET_H */
