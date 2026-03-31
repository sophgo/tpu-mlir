#include <stddef.h>
#include <stdint.h>

typedef enum {
  CN_DTYPE_FLOAT,
  CN_DTYPE_INT,
  CN_DTYPE_UINT,
  CN_DTYPE_BOOL,
  CN_DTYPE_COMPLEX
} cn_dtype_class_t;

int cnpy_npz_add(const char *zipname,
                 const char *varname,
                 const void *data_ptr,
                 size_t elem_size,
                 size_t elem_count,
                 const char *shape_str,
                 cn_dtype_class_t dtype_class,
                 const char *mode);

int cnpy_build_shape(char *out, size_t out_sz, const size_t *dims, size_t ndims);
