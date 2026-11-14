/*
 * torchnative's shim over the ExecuTorch v1.4.1 runtime (issue #315, step R1).
 *
 * The vendored runtime (torchnative/rust/vendor/executorch) is compiled by
 * ../build.rs into one static archive together with this file. This file is
 * the only one that knows C++ types; tn_et.h is the whole surface Rust sees.
 *
 * What it adds on top of examples/portable/executor_runner's shape:
 *
 *  - PAL hooks through register_pal(): the runtime's log lines go into a
 *    per-thread buffer, so a failed call's exception can carry the line where
 *    ExecuTorch named the reason ("Missing operator: ...", "Verification
 *    failed ..."); and an abort prints that buffer before the process dies,
 *    because an abort is the one failure this shim cannot turn into a status.
 *  - Program::Verification::InternalConsistency on every load, plus a size
 *    pre-check, so a truncated or garbage buffer returns InvalidProgram
 *    instead of reaching the flatbuffer accessors (which would read out of
 *    bounds or ET_CHECK).
 *  - Input validation against MethodMeta before Method::set_input, so a wrong
 *    dtype, rank or byte count returns InvalidArgument with a message, and
 *    elementSize() (which ET_CHECKs on an unknown ScalarType) is never reached
 *    with a code outside the allow-list.
 *  - A counting EventTracer: the generated kernel wrappers open a profiling
 *    scope named "native_call_<op>.<overload>" around every kernel call
 *    (RegisterCodegenUnboxedKernelsEverything.cpp), which is runtime
 *    evidence of which kernel ran, not a description of the program.
 *
 * Built with -fno-exceptions -fno-rtti like upstream: nothing here throws,
 * allocation goes through std::nothrow or the runtime's allocators, and a
 * failure is a returned status.
 */
#include "tn_et.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/extended_header.h>

namespace et = ::executorch::runtime;
namespace etr = ::executorch::ET_RUNTIME_NAMESPACE;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;
using ::executorch::extension::BufferDataLoader;
using ::executorch::extension::MallocMemoryAllocator;
using et::Error;
using et::EValue;
using et::Result;
using et::Span;
using et::Tag;

namespace {

constexpr int32_t kOk = static_cast<int32_t>(Error::Ok);

int32_t code(Error e) {
  return static_cast<int32_t>(e);
}

// ---------------------------------------------------------------- the log

// Per thread, fixed size, trivially constructible (no TLS destructor). The
// runtime logs on the thread that called it (no threadpool in R1), so the
// lines in here belong to the call that just failed on this thread.
constexpr size_t kLogCap = 8192;
thread_local char tl_log[kLogCap];
thread_local size_t tl_log_len = 0;

void log_clear() {
  tl_log_len = 0;
  tl_log[0] = '\0';
}

void log_append(const char* text, size_t n) {
  if (n >= kLogCap) {
    text += n - (kLogCap - 1);
    n = kLogCap - 1;
  }
  if (tl_log_len + n >= kLogCap) {
    // Keep the newest lines: drop from the front.
    size_t drop = tl_log_len + n - (kLogCap - 1);
    std::memmove(tl_log, tl_log + drop, tl_log_len - drop);
    tl_log_len -= drop;
  }
  std::memcpy(tl_log + tl_log_len, text, n);
  tl_log_len += n;
  tl_log[tl_log_len] = '\0';
}

const char* base_name(const char* path) {
  if (path == nullptr) {
    return "?";
  }
  const char* slash = std::strrchr(path, '/');
  return slash != nullptr ? slash + 1 : path;
}

// The shim's own diagnostics go into the same buffer, marked 'S'.
void note(const char* fmt, ...) {
  char line[1024];
  int head = std::snprintf(line, sizeof(line), "S tn_et] ");
  va_list args;
  va_start(args, fmt);
  int body = std::vsnprintf(line + head, sizeof(line) - head, fmt, args);
  va_end(args);
  size_t n = static_cast<size_t>(head) +
      (body < 0 ? 0 : static_cast<size_t>(body));
  if (n > sizeof(line) - 2) {
    n = sizeof(line) - 2;
  }
  line[n++] = '\n';
  log_append(line, n);
}

void emit_log(
    et_timestamp_t /*timestamp*/,
    et_pal_log_level_t level,
    const char* filename,
    const char* /*function*/,
    size_t line,
    const char* message,
    size_t length) {
  char head[160];
  int h = std::snprintf(
      head, sizeof(head), "%c %s:%zu] ", static_cast<char>(level),
      base_name(filename), line);
  if (h > 0) {
    log_append(head, static_cast<size_t>(h) < sizeof(head) ? h : sizeof(head) - 1);
  }
  size_t n = message != nullptr ? strnlen(message, length) : 0;
  log_append(message != nullptr ? message : "", n);
  log_append("\n", 1);
}

// ET_CHECK and friends end here. It cannot return (the runtime would carry
// on with a violated invariant), and it cannot unwind (no exceptions, and
// unwinding through Rust frames is undefined). So the only useful thing left
// is to say what happened before the process dies.
[[noreturn]] void report_abort() {
  std::fprintf(
      stderr,
      "torchnative executorch_rt: the ExecuTorch runtime aborted the process "
      "(an ET_CHECK failed or et_pal_abort was called). This is not a Python "
      "exception and cannot be caught. Last runtime log on this thread:\n%s\n",
      tl_log_len != 0 ? tl_log : "(empty)");
  std::fflush(stderr);
  std::abort();
}

bool init_once() {
  et::register_pal(et::PalImpl::create(
      /*init=*/nullptr,
      /*abort=*/&report_abort,
      /*current_ticks=*/nullptr,
      /*ticks_to_ns_multiplier=*/nullptr,
      /*emit_log_message=*/&emit_log,
      /*allocate=*/nullptr,
      /*free=*/nullptr,
      __FILE__));
  et::runtime_init();
  return true;
}

void ensure_init() {
  // A function-local static is initialized once, thread-safely (C++11 magic
  // statics; no exceptions involved).
  static const bool done = init_once();
  (void)done;
}

// ------------------------------------------------------------ dtype rules

// The ScalarType codes a caller may pass. elementSize() and toString()
// ET_CHECK on anything outside the enum, so the shim never forwards a code
// it has not checked here.
bool known_scalar_type(int8_t st) {
  switch (st) {
    case 0:  // Byte
    case 1:  // Char
    case 2:  // Short
    case 3:  // Int
    case 4:  // Long
    case 5:  // Half
    case 6:  // Float
    case 7:  // Double
    case 11: // Bool
    case 15: // BFloat16
    case 27: // UInt16
    case 28: // UInt32
    case 29: // UInt64
      return true;
    default:
      return false;
  }
}

// ------------------------------------------------------ the counting tracer

class CountingTracer final : public et::EventTracer {
 public:
  std::map<std::string, uint64_t> operator_calls;
  std::map<std::string, uint64_t> scopes;
  std::map<std::string, uint64_t> delegate_ops;

  void reset() {
    operator_calls.clear();
    scopes.clear();
    delegate_ops.clear();
  }

  void create_event_block(const char* /*name*/) override {}

  et::EventTracerEntry start_profiling(
      const char* name,
      et::ChainID chain_id,
      et::DebugHandle debug_handle) override {
    static const char kPrefix[] = "native_call_";
    constexpr size_t kPrefixLen = sizeof(kPrefix) - 1;
    if (name != nullptr) {
      scopes[name] += 1;
      if (std::strncmp(name, kPrefix, kPrefixLen) == 0) {
        operator_calls[name + kPrefixLen] += 1;
      }
    }
    return entry(chain_id, debug_handle);
  }

  et::EventTracerEntry start_profiling_delegate(
      const char* name,
      et::DelegateDebugIntId /*delegate_debug_index*/) override {
    scopes[std::string("delegate:") + (name != nullptr ? name : "?")] += 1;
    return entry(et::kUnsetChainId, et::kUnsetDebugHandle);
  }

  void end_profiling_delegate(
      et::EventTracerEntry /*event_tracer_entry*/,
      const void* /*metadata*/,
      size_t /*metadata_len*/) override {}

  void log_profiling_delegate(
      const char* name,
      et::DelegateDebugIntId /*delegate_debug_index*/,
      et_timestamp_t /*start_time*/,
      et_timestamp_t /*end_time*/,
      const void* /*metadata*/,
      size_t /*metadata_len*/) override {
    delegate_ops[name != nullptr ? name : "?"] += 1;
  }

  void end_profiling(et::EventTracerEntry /*prof_entry*/) override {}

  void track_allocation(et::AllocatorID /*id*/, size_t /*size*/) override {}

  et::AllocatorID track_allocator(const char* /*name*/) override {
    return 0;
  }

  Result<bool> log_evalue(
      const EValue& /*evalue*/,
      et::LoggedEValueType /*evalue_type*/) override {
    return Result<bool>(false);
  }

  Result<bool> log_intermediate_output_delegate(
      const char*,
      et::DelegateDebugIntId,
      const Tensor&) override {
    return Result<bool>(false);
  }

  Result<bool> log_intermediate_output_delegate(
      const char*,
      et::DelegateDebugIntId,
      const et::ArrayRef<Tensor>) override {
    return Result<bool>(false);
  }

  Result<bool> log_intermediate_output_delegate(
      const char*,
      et::DelegateDebugIntId,
      const int&) override {
    return Result<bool>(false);
  }

  Result<bool> log_intermediate_output_delegate(
      const char*,
      et::DelegateDebugIntId,
      const bool&) override {
    return Result<bool>(false);
  }

  Result<bool> log_intermediate_output_delegate(
      const char*,
      et::DelegateDebugIntId,
      const double&) override {
    return Result<bool>(false);
  }

  void set_delegation_intermediate_output_filter(
      et::EventTracerFilterBase* /*event_tracer_filter*/) override {}

 private:
  int64_t next_id_ = 0;

  et::EventTracerEntry entry(et::ChainID chain_id, et::DebugHandle handle) {
    et::EventTracerEntry e;
    e.event_id = next_id_++;
    e.chain_id = chain_id;
    e.debug_handle = handle;
    e.start_time = 0;
    e.delegate_event_id_type = et::DelegateDebugIdType::kNone;
    return e;
  }
};

const std::map<std::string, uint64_t>* family_map(
    const CountingTracer& tracer,
    int32_t family) {
  switch (family) {
    case TN_ET_COUNT_OPERATOR_CALLS:
      return &tracer.operator_calls;
    case TN_ET_COUNT_SCOPES:
      return &tracer.scopes;
    case TN_ET_COUNT_DELEGATE_OPS:
      return &tracer.delegate_ops;
    default:
      return nullptr;
  }
}

// ---------------------------------------------------------- aligned bytes

constexpr size_t kArenaAlignment = 64;

void* aligned_bytes(size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, kArenaAlignment, n == 0 ? 1 : n) != 0) {
    return nullptr;
  }
  return p;
}

void fill_tensor_info(const etr::TensorInfo& info, tn_et_tensor_info* out) {
  out->tag = TN_ET_TAG_TENSOR;
  out->scalar_type = static_cast<int8_t>(info.scalar_type());
  out->memory_planned = info.is_memory_planned() ? 1 : 0;
  out->ndim = info.sizes().size();
  out->sizes = info.sizes().data();
  out->dim_order = info.dim_order().data();
  out->nbytes = info.nbytes();
}

int32_t io_meta(
    const etr::MethodMeta& meta,
    bool input,
    size_t index,
    tn_et_tensor_info* out) {
  std::memset(out, 0, sizeof(*out));
  size_t n = input ? meta.num_inputs() : meta.num_outputs();
  if (index >= n) {
    note("%s index %zu out of range: method has %zu",
         input ? "input" : "output", index, n);
    return code(Error::InvalidArgument);
  }
  Result<Tag> tag = input ? meta.input_tag(index) : meta.output_tag(index);
  if (!tag.ok()) {
    return code(tag.error());
  }
  out->tag = static_cast<int32_t>(tag.get());
  if (tag.get() != Tag::Tensor) {
    return kOk;
  }
  Result<etr::TensorInfo> info =
      input ? meta.input_tensor_meta(index) : meta.output_tensor_meta(index);
  if (!info.ok()) {
    return code(info.error());
  }
  fill_tensor_info(info.get(), out);
  return kOk;
}

} // namespace

// ------------------------------------------------------------------ handles

struct tn_et_program {
  BufferDataLoader loader;
  etr::Program* program = nullptr;

  tn_et_program(const void* data, size_t len) : loader(data, len) {}
  ~tn_et_program() {
    delete program;
  }
};

struct tn_et_method {
  // Declaration order is destruction order reversed; ~tn_et_method frees the
  // Method first explicitly, because it is the one object that points into
  // everything else (allocators, planned arenas, tracer).
  MallocMemoryAllocator method_allocator;
  MallocMemoryAllocator temp_allocator;
  std::vector<void*> planned_raw;
  std::vector<Span<uint8_t>> planned_spans;
  std::unique_ptr<et::HierarchicalAllocator> planned;
  std::unique_ptr<et::MemoryManager> manager;
  CountingTracer tracer;
  etr::Method* method = nullptr;
  // Owned copies for inputs the program did not memory-plan: Method shares
  // the pointer for those (method.cpp, share_tensor_data), so it must outlive
  // the call that set it. Planned inputs are copied into the arena instead.
  std::vector<void*> input_raw;
  std::vector<size_t> input_cap;
  uint64_t executions = 0;

  ~tn_et_method() {
    delete method;
    method = nullptr;
    manager.reset();
    planned.reset();
    for (void* p : planned_raw) {
      std::free(p);
    }
    for (void* p : input_raw) {
      std::free(p);
    }
  }
};

// --------------------------------------------------------------- functions

extern "C" {

int32_t tn_et_init(void) {
  ensure_init();
  return kOk;
}

size_t tn_et_take_log(char* buf, size_t cap) {
  size_t n = tl_log_len;
  if (buf != nullptr && cap > 0) {
    size_t c = n < cap - 1 ? n : cap - 1;
    std::memcpy(buf, tl_log, c);
    buf[c] = '\0';
  }
  log_clear();
  return n;
}

int32_t tn_et_event_tracer_enabled(void) {
#ifdef ET_EVENT_TRACER_ENABLED
  return 1;
#else
  return 0;
#endif
}

int32_t tn_et_program_verification_enabled(void) {
#if defined(ET_ENABLE_PROGRAM_VERIFICATION) && ET_ENABLE_PROGRAM_VERIFICATION
  return 1;
#else
  return 0;
#endif
}

size_t tn_et_kernel_count(void) {
  ensure_init();
  return etr::get_registered_kernels().size();
}

const char* tn_et_kernel_name(size_t index) {
  ensure_init();
  Span<const etr::Kernel> kernels = etr::get_registered_kernels();
  if (index >= kernels.size()) {
    return nullptr;
  }
  return kernels[index].name_;
}

size_t tn_et_backend_count(void) {
  ensure_init();
  return etr::get_num_registered_backends();
}

const char* tn_et_backend_name(size_t index) {
  ensure_init();
  Result<const char*> name = etr::get_backend_name(index);
  return name.ok() ? name.get() : nullptr;
}

int32_t tn_et_program_load(const uint8_t* data, size_t len, tn_et_program** out) {
  ensure_init();
  log_clear();
  if (out == nullptr) {
    return code(Error::InvalidArgument);
  }
  *out = nullptr;
  if (data == nullptr) {
    note("program buffer is NULL");
    return code(Error::InvalidArgument);
  }
  // Program::load first asks the loader for the 64-byte head, and a shorter
  // buffer makes the loader answer InvalidArgument. A buffer that short is
  // not a program, which is what the caller needs to hear.
  if (len < ::executorch::runtime::ExtendedHeader::kNumHeadBytes) {
    note("program buffer is %zu bytes; a program is at least %zu",
         len, static_cast<size_t>(::executorch::runtime::ExtendedHeader::kNumHeadBytes));
    return code(Error::InvalidProgram);
  }
  tn_et_program* handle = new (std::nothrow) tn_et_program(data, len);
  if (handle == nullptr) {
    return code(Error::MemoryAllocationFailed);
  }
  Result<etr::Program> loaded = etr::Program::load(
      &handle->loader, etr::Program::Verification::InternalConsistency);
  if (!loaded.ok()) {
    Error e = loaded.error();
    delete handle;
    return code(e);
  }
  handle->program = new (std::nothrow) etr::Program(std::move(loaded.get()));
  if (handle->program == nullptr) {
    delete handle;
    return code(Error::MemoryAllocationFailed);
  }
  *out = handle;
  return kOk;
}

void tn_et_program_free(tn_et_program* program) {
  delete program;
}

size_t tn_et_program_num_methods(const tn_et_program* program) {
  return program != nullptr ? program->program->num_methods() : 0;
}

const char* tn_et_program_method_name(const tn_et_program* program, size_t index) {
  if (program == nullptr || index >= program->program->num_methods()) {
    return nullptr;
  }
  Result<const char*> name = program->program->get_method_name(index);
  return name.ok() ? name.get() : nullptr;
}

int32_t tn_et_method_meta(
    const tn_et_program* program,
    const char* name,
    tn_et_method_info* out) {
  log_clear();
  if (program == nullptr || name == nullptr || out == nullptr) {
    return code(Error::InvalidArgument);
  }
  Result<etr::MethodMeta> meta = program->program->method_meta(name);
  if (!meta.ok()) {
    return code(meta.error());
  }
  out->num_inputs = meta->num_inputs();
  out->num_outputs = meta->num_outputs();
  out->num_planned_buffers = meta->num_memory_planned_buffers();
  out->num_backends = meta->num_backends();
  out->num_instructions = meta->num_instructions();
  return kOk;
}

int32_t tn_et_method_input_meta(
    const tn_et_program* program,
    const char* name,
    size_t index,
    tn_et_tensor_info* out) {
  log_clear();
  if (program == nullptr || name == nullptr || out == nullptr) {
    return code(Error::InvalidArgument);
  }
  Result<etr::MethodMeta> meta = program->program->method_meta(name);
  if (!meta.ok()) {
    return code(meta.error());
  }
  return io_meta(meta.get(), /*input=*/true, index, out);
}

int32_t tn_et_method_output_meta(
    const tn_et_program* program,
    const char* name,
    size_t index,
    tn_et_tensor_info* out) {
  log_clear();
  if (program == nullptr || name == nullptr || out == nullptr) {
    return code(Error::InvalidArgument);
  }
  Result<etr::MethodMeta> meta = program->program->method_meta(name);
  if (!meta.ok()) {
    return code(meta.error());
  }
  return io_meta(meta.get(), /*input=*/false, index, out);
}

int32_t tn_et_method_planned_buffer_size(
    const tn_et_program* program,
    const char* name,
    size_t index,
    int64_t* out) {
  log_clear();
  if (program == nullptr || name == nullptr || out == nullptr) {
    return code(Error::InvalidArgument);
  }
  Result<etr::MethodMeta> meta = program->program->method_meta(name);
  if (!meta.ok()) {
    return code(meta.error());
  }
  Result<int64_t> size = meta->memory_planned_buffer_size(index);
  if (!size.ok()) {
    return code(size.error());
  }
  *out = size.get();
  return kOk;
}

int32_t tn_et_method_backend_name(
    const tn_et_program* program,
    const char* name,
    size_t index,
    const char** out) {
  log_clear();
  if (program == nullptr || name == nullptr || out == nullptr) {
    return code(Error::InvalidArgument);
  }
  Result<etr::MethodMeta> meta = program->program->method_meta(name);
  if (!meta.ok()) {
    return code(meta.error());
  }
  Result<const char*> backend = meta->get_backend_name(index);
  if (!backend.ok()) {
    return code(backend.error());
  }
  *out = backend.get();
  return kOk;
}

int32_t tn_et_method_load(
    const tn_et_program* program,
    const char* name,
    tn_et_method** out) {
  ensure_init();
  log_clear();
  if (program == nullptr || name == nullptr || out == nullptr) {
    return code(Error::InvalidArgument);
  }
  *out = nullptr;
  Result<etr::MethodMeta> meta = program->program->method_meta(name);
  if (!meta.ok()) {
    return code(meta.error());
  }
  std::unique_ptr<tn_et_method> handle(new (std::nothrow) tn_et_method());
  if (!handle) {
    return code(Error::MemoryAllocationFailed);
  }

  // Memory planning: one arena per planned buffer, sized by the program's
  // own MethodMeta, owned by the handle (examples/portable/executor_runner
  // does the same with make_unique<uint8_t[]>).
  size_t n_planned = meta->num_memory_planned_buffers();
  handle->planned_raw.reserve(n_planned);
  handle->planned_spans.reserve(n_planned);
  for (size_t i = 0; i < n_planned; ++i) {
    Result<int64_t> size = meta->memory_planned_buffer_size(i);
    if (!size.ok()) {
      return code(size.error());
    }
    if (size.get() < 0) {
      note("planned buffer %zu has negative size %lld", i,
           static_cast<long long>(size.get()));
      return code(Error::InvalidProgram);
    }
    size_t bytes = static_cast<size_t>(size.get());
    void* raw = aligned_bytes(bytes);
    if (raw == nullptr) {
      note("cannot allocate planned buffer %zu of %zu bytes", i, bytes);
      return code(Error::MemoryAllocationFailed);
    }
    handle->planned_raw.push_back(raw);
    handle->planned_spans.push_back(
        Span<uint8_t>(static_cast<uint8_t*>(raw), bytes));
  }
  handle->planned.reset(new (std::nothrow) et::HierarchicalAllocator(
      Span<Span<uint8_t>>(handle->planned_spans.data(), handle->planned_spans.size())));
  if (!handle->planned) {
    return code(Error::MemoryAllocationFailed);
  }
  handle->manager.reset(new (std::nothrow) et::MemoryManager(
      &handle->method_allocator, handle->planned.get(), &handle->temp_allocator));
  if (!handle->manager) {
    return code(Error::MemoryAllocationFailed);
  }

  Result<etr::Method> loaded = program->program->load_method(
      name, handle->manager.get(), &handle->tracer);
  if (!loaded.ok()) {
    return code(loaded.error());
  }
  handle->method = new (std::nothrow) etr::Method(std::move(loaded.get()));
  if (handle->method == nullptr) {
    return code(Error::MemoryAllocationFailed);
  }
  size_t n_inputs = handle->method->inputs_size();
  handle->input_raw.assign(n_inputs, nullptr);
  handle->input_cap.assign(n_inputs, 0);
  // Method::init ran kernels-resolution scopes through the tracer; counters
  // start from the first execute.
  handle->tracer.reset();
  *out = handle.release();
  return kOk;
}

void tn_et_method_free(tn_et_method* method) {
  delete method;
}

size_t tn_et_method_num_inputs(const tn_et_method* method) {
  return method != nullptr ? method->method->inputs_size() : 0;
}

size_t tn_et_method_num_outputs(const tn_et_method* method) {
  return method != nullptr ? method->method->outputs_size() : 0;
}

int32_t tn_et_method_set_input(
    tn_et_method* method,
    size_t index,
    int8_t scalar_type,
    const int64_t* sizes,
    size_t ndim,
    const void* data,
    size_t nbytes) {
  log_clear();
  if (method == nullptr || (ndim > 0 && sizes == nullptr) ||
      (nbytes > 0 && data == nullptr)) {
    return code(Error::InvalidArgument);
  }
  etr::MethodMeta meta = method->method->method_meta();
  if (index >= meta.num_inputs()) {
    note("input index %zu out of range: the method takes %zu inputs", index,
         meta.num_inputs());
    return code(Error::InvalidArgument);
  }
  Result<Tag> tag = meta.input_tag(index);
  if (!tag.ok()) {
    return code(tag.error());
  }
  if (tag.get() != Tag::Tensor) {
    note("input %zu is not a tensor in this program (tag %d); the binding "
         "passes tensors only", index, static_cast<int>(tag.get()));
    return code(Error::NotSupported);
  }
  Result<etr::TensorInfo> info = meta.input_tensor_meta(index);
  if (!info.ok()) {
    return code(info.error());
  }
  if (!known_scalar_type(scalar_type)) {
    note("input %zu: scalar type code %d is not one the binding passes", index,
         static_cast<int>(scalar_type));
    return code(Error::InvalidArgument);
  }
  ScalarType want = info->scalar_type();
  ScalarType got = static_cast<ScalarType>(scalar_type);
  if (want != got) {
    note("input %zu: the program expects %s, got %s", index,
         et::toString(want), et::toString(got));
    return code(Error::InvalidArgument);
  }
  if (ndim != info->sizes().size()) {
    note("input %zu: the program expects rank %zu, got rank %zu", index,
         info->sizes().size(), ndim);
    return code(Error::InvalidArgument);
  }
  if (ndim > ::executorch::runtime::kTensorDimensionLimit) {
    note("input %zu: rank %zu exceeds the runtime's limit %zu", index, ndim,
         static_cast<size_t>(::executorch::runtime::kTensorDimensionLimit));
    return code(Error::InvalidArgument);
  }
  TensorImpl::SizesType sizes32[::executorch::runtime::kTensorDimensionLimit];
  TensorImpl::DimOrderType order[::executorch::runtime::kTensorDimensionLimit];
  TensorImpl::StridesType strides[::executorch::runtime::kTensorDimensionLimit];
  size_t numel = 1;
  for (size_t d = 0; d < ndim; ++d) {
    if (sizes[d] < 0 || sizes[d] > INT32_MAX) {
      note("input %zu: size %lld at dim %zu is out of range", index,
           static_cast<long long>(sizes[d]), d);
      return code(Error::InvalidArgument);
    }
    if (sizes[d] != info->sizes()[d]) {
      // R1 programs are static; a dynamic-shape program declares its upper
      // bound here and would need resize support this round does not claim.
      note("input %zu: the program expects size %d at dim %zu, got %lld", index,
           static_cast<int>(info->sizes()[d]), d, static_cast<long long>(sizes[d]));
      return code(Error::InvalidArgument);
    }
    sizes32[d] = static_cast<TensorImpl::SizesType>(sizes[d]);
    order[d] = static_cast<TensorImpl::DimOrderType>(d);
    size_t next = 0;
    if (__builtin_mul_overflow(numel, static_cast<size_t>(sizes[d]), &next)) {
      note("input %zu: element count overflows", index);
      return code(Error::InvalidArgument);
    }
    numel = next;
  }
  {
    // Contiguous strides, innermost last.
    int64_t stride = 1;
    for (size_t d = ndim; d-- > 0;) {
      strides[d] = static_cast<TensorImpl::StridesType>(stride);
      stride *= sizes32[d];
    }
  }
  size_t want_bytes = 0;
  if (__builtin_mul_overflow(numel, et::elementSize(got), &want_bytes) ||
      want_bytes != nbytes) {
    note("input %zu: %zu bytes passed, the shape and dtype need %zu", index,
         nbytes, want_bytes);
    return code(Error::InvalidArgument);
  }

  void* source = const_cast<void*>(data);
  if (!info->is_memory_planned()) {
    // The runtime will keep this pointer, so it must be ours, not the
    // caller's: copy into a buffer the handle owns (still one copy).
    if (method->input_cap[index] < nbytes || method->input_raw[index] == nullptr) {
      void* fresh = aligned_bytes(nbytes);
      if (fresh == nullptr) {
        note("input %zu: cannot allocate %zu bytes", index, nbytes);
        return code(Error::MemoryAllocationFailed);
      }
      std::free(method->input_raw[index]);
      method->input_raw[index] = fresh;
      method->input_cap[index] = nbytes;
    }
    if (nbytes > 0) {
      std::memcpy(method->input_raw[index], data, nbytes);
    }
    source = method->input_raw[index];
  }
  TensorImpl impl(
      got,
      static_cast<ssize_t>(ndim),
      sizes32,
      source,
      order,
      strides,
      ::executorch::aten::TensorShapeDynamism::STATIC);
  EValue value{Tensor(&impl)};
  return code(method->method->set_input(value, index));
}

int32_t tn_et_method_execute(tn_et_method* method) {
  log_clear();
  if (method == nullptr) {
    return code(Error::InvalidArgument);
  }
  Error e = method->method->execute();
  if (e == Error::Ok) {
    method->executions += 1;
  }
  return code(e);
}

int32_t tn_et_method_output(
    const tn_et_method* method,
    size_t index,
    tn_et_value* out) {
  log_clear();
  if (method == nullptr || out == nullptr) {
    return code(Error::InvalidArgument);
  }
  std::memset(out, 0, sizeof(*out));
  if (index >= method->method->outputs_size()) {
    // Method::get_output ET_CHECKs the index; refuse before it.
    note("output index %zu out of range: the method has %zu outputs", index,
         method->method->outputs_size());
    return code(Error::InvalidArgument);
  }
  const EValue& value = method->method->get_output(index);
  out->tag = static_cast<int32_t>(value.tag);
  switch (value.tag) {
    case Tag::Tensor: {
      const Tensor& t = value.toTensor();
      out->scalar_type = static_cast<int8_t>(t.scalar_type());
      out->ndim = static_cast<size_t>(t.dim());
      out->sizes = t.sizes().data();
      out->dim_order = t.dim_order().data();
      out->data = t.const_data_ptr();
      out->nbytes = t.nbytes();
      return kOk;
    }
    case Tag::Int:
      out->int_value = value.toInt();
      return kOk;
    case Tag::Double:
      out->double_value = value.toDouble();
      return kOk;
    case Tag::Bool:
      out->bool_value = value.toBool() ? 1 : 0;
      return kOk;
    case Tag::None:
      return kOk;
    default:
      note("output %zu has tag %d, which the binding does not return", index,
           static_cast<int>(value.tag));
      return code(Error::NotSupported);
  }
}

uint64_t tn_et_method_executions(const tn_et_method* method) {
  return method != nullptr ? method->executions : 0;
}

size_t tn_et_method_counter_count(const tn_et_method* method, int32_t family) {
  if (method == nullptr) {
    return 0;
  }
  const std::map<std::string, uint64_t>* map = family_map(method->tracer, family);
  return map != nullptr ? map->size() : 0;
}

int32_t tn_et_method_counter_at(
    const tn_et_method* method,
    int32_t family,
    size_t index,
    const char** name,
    uint64_t* count) {
  if (method == nullptr || name == nullptr || count == nullptr) {
    return code(Error::InvalidArgument);
  }
  const std::map<std::string, uint64_t>* map = family_map(method->tracer, family);
  if (map == nullptr || index >= map->size()) {
    return code(Error::InvalidArgument);
  }
  auto it = map->begin();
  std::advance(it, static_cast<long>(index));
  *name = it->first.c_str();
  *count = it->second;
  return kOk;
}

void tn_et_method_reset_counters(tn_et_method* method) {
  if (method != nullptr) {
    method->tracer.reset();
    method->executions = 0;
  }
}

} // extern "C"
