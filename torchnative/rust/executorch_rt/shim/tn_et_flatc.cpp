/*
 * flatc's JSON-to-binary step, linked instead of exec'd (issue #357; leader
 * decision 2026-11-14 item 4: "flatbuffers C++ parser linked into the
 * extension, replacing exir's _run_flatc in process; _get_flatc_path refuses
 * by name").
 *
 * The one mode exir and the XNNPACK serializer use for an XNNPACK lowering,
 * measured on executorch 1.4.1 (exir/_serialize/_flatbuffer.py
 * `_flatc_compile`, called from
 * backends/xnnpack/serialization/xnnpack_graph_serialize.py
 * `convert_to_flatbuffer`):
 *
 *     flatc --binary -o DIR DIR/schema.fbs DIR/schema.json
 *
 * flatc 24.3.25 (src/flatc.cpp) then does, and this file does the same, in
 * the same order, with the same parser code (the vendored v24.3.25 sources):
 *
 *   1. IDLOptions at their defaults (`--binary` sets no parser option; it
 *      selects the binary generator, recorded in lang_to_generate);
 *   2. refuse a file whose text holds a NUL ("input file appears to be
 *      binary");
 *   3. a fresh Parser, Parse(schema);
 *   4. Parse(json) on that same parser;
 *   5. refuse an empty builder ("input file is neither json nor a .fbs
 *      (schema) file");
 *   6. the binary generator writes builder_'s bytes as they are; here they
 *      are returned.
 *
 * What differs from flatc, on purpose: no file is touched. The parser's
 * file hooks (util.h SetLoadFileFunction / SetFileExistsFunction) are pointed
 * at functions that find nothing for the duration of the call, so an
 * `include` in a schema fails with the parser's own "unable to locate include
 * file" instead of reading the disk (no measured schema has one), and the
 * flatbuffers sources are compiled with FLATBUFFERS_NO_ABSOLUTE_PATH_RESOLUTION
 * so the parser never calls realpath. Those hooks are process-global, hence
 * the mutex; nothing else in this extension uses flatbuffers' file functions.
 *
 * Errors: the parser reports malformed input through Parse() == false and
 * error_, never by asserting (its asserts guard internal invariants). This
 * translation unit and the flatbuffers sources are compiled with exceptions
 * on, so an allocation failure inside the parser (std::bad_alloc) becomes a
 * returned status here rather than std::terminate, and no exception crosses
 * the extern "C" boundary.
 */
#include "tn_et.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <string>

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

namespace {

std::mutex g_flatc_mutex;

bool find_no_file(const char* /*name*/) {
  return false;
}

bool load_no_file(const char* /*name*/, bool /*binary*/, std::string* /*dest*/) {
  return false;
}

/* Points the parser's file hooks at find_no_file/load_no_file and restores
 * the previous ones when the call ends. */
class NoFiles {
 public:
  NoFiles()
      : load_(flatbuffers::SetLoadFileFunction(load_no_file)),
        exists_(flatbuffers::SetFileExistsFunction(find_no_file)) {}
  ~NoFiles() {
    flatbuffers::SetLoadFileFunction(load_);
    flatbuffers::SetFileExistsFunction(exists_);
  }
  NoFiles(const NoFiles&) = delete;
  NoFiles& operator=(const NoFiles&) = delete;

 private:
  flatbuffers::LoadFileFunction load_;
  flatbuffers::FileExistsFunction exists_;
};

char* copy_message(const std::string& message) {
  char* out = static_cast<char*>(std::malloc(message.size() + 1));
  if (out != nullptr) {
    std::memcpy(out, message.data(), message.size());
    out[message.size()] = '\0';
  }
  return out;
}

int32_t fail(char** error, const std::string& message) {
  if (error != nullptr) {
    *error = copy_message(message);
  }
  return TN_ET_STATUS_FLATC_ERROR;
}

std::string name_or(const char* name, const char* fallback) {
  return name != nullptr && name[0] != '\0' ? std::string(name) : std::string(fallback);
}

int32_t convert(const char* schema, size_t schema_len, const char* schema_name,
                const char* json, size_t json_len, const char* json_name,
                uint8_t** out, size_t* out_len, char** error) {
  const std::string schema_file = name_or(schema_name, "schema.fbs");
  const std::string json_file = name_or(json_name, "data.json");
  // flatc reads each file into a std::string and refuses one whose length
  // differs from strlen (a NUL inside): the parser works on C strings.
  const std::string schema_text(schema, schema_len);
  const std::string json_text(json, json_len);
  if (schema_text.size() != std::strlen(schema_text.c_str())) {
    return fail(error, "input file appears to be binary: " + schema_file);
  }
  if (json_text.size() != std::strlen(json_text.c_str())) {
    return fail(error, "input file appears to be binary: " + json_file);
  }

  std::lock_guard<std::mutex> lock(g_flatc_mutex);
  NoFiles no_files;

  flatbuffers::IDLOptions opts;
  opts.lang_to_generate = flatbuffers::IDLOptions::kBinary;
  flatbuffers::Parser parser(opts);
  if (!parser.Parse(schema_text.c_str(), nullptr, schema_file.c_str())) {
    return fail(error, parser.error_);
  }
  if (!parser.Parse(json_text.c_str(), nullptr, json_file.c_str())) {
    return fail(error, parser.error_);
  }
  const size_t size = parser.builder_.GetSize();
  if (size == 0) {
    return fail(error, "input file is neither json nor a .fbs (schema) file: " + json_file);
  }
  uint8_t* bytes = static_cast<uint8_t*>(std::malloc(size));
  if (bytes == nullptr) {
    return fail(error, "out of memory copying " + std::to_string(size) + " bytes of flatbuffer output");
  }
  std::memcpy(bytes, parser.builder_.GetBufferPointer(), size);
  *out = bytes;
  *out_len = size;
  return 0;
}

}  // namespace

extern "C" int32_t tn_et_flatc_json_to_binary(const char* schema, size_t schema_len,
                                              const char* schema_name, const char* json,
                                              size_t json_len, const char* json_name,
                                              uint8_t** out, size_t* out_len,
                                              char** error) {
  if (out == nullptr || out_len == nullptr || error == nullptr) {
    return TN_ET_STATUS_FLATC_ERROR;
  }
  *out = nullptr;
  *out_len = 0;
  *error = nullptr;
  if ((schema == nullptr && schema_len != 0) || (json == nullptr && json_len != 0)) {
    return fail(error, "tn_et_flatc_json_to_binary: null text with a non-zero length");
  }
  try {
    return convert(schema != nullptr ? schema : "", schema_len, schema_name,
                   json != nullptr ? json : "", json_len, json_name, out, out_len, error);
  } catch (const std::bad_alloc&) {
    return fail(error, "out of memory inside the flatbuffers parser");
  } catch (const std::exception& e) {
    return fail(error, std::string("the flatbuffers parser threw: ") + e.what());
  } catch (...) {
    return fail(error, "the flatbuffers parser threw a non-standard exception");
  }
}

extern "C" void tn_et_flatc_free(void* ptr) {
  std::free(ptr);
}
