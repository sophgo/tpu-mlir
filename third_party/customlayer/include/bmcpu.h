#ifndef _CPU_OP_H_
#define _CPU_OP_H_

#include <vector>
#include <string>
using std::vector;
using std::string;

#ifdef _WIN32
#define DECL_EXPORT _declspec(dllexport)
#define DECL_IMPORT _declspec(dllimport)
#else
#define DECL_EXPORT
#define DECL_IMPORT
#endif

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @name   bmcpu_init
 * @brief  initialize bmcpu library
 *
 * @retval  bmcpu handler
 */
DECL_EXPORT void* bmcpu_init();

/**
 * @name   bmcpu_uninit
 * @brief  deinitialize bmcpu library
 *
 * @param   [in]    bmcpu_handle  The pointer of cpu handler.
 */
DECL_EXPORT void bmcpu_uninit(void* bmcpu_handle);

/**
 * @name    customcpu_process
 * @brief   Call cpu process
 *
 * The interface will call the process the corresponding cpu layer
 *
 * @param   [in]    bmcpu_handle   The pointer of cpu handler.
 * @param   [in]    op_type        The type of the cpu op that is defined in CPU_LAYER_TYPE.
 * @param   [in]    param          The pointer of the cpu op parameter.
 * @param   [in]    param_size     The byte size of the parameter.
 * @param   [in]    input_tensors  The data pointer of each inpyyut tensor.
 * @param   [in]    input_shapes   The shape of each input tensor.
 * @param   [in]    output_tensors The data pointer of each output tensor.
 * @param   [in]    output_shapes  The shape of each output tensor.
 *
 * @retval  0      success
 * @retval  other  fail
 */
DECL_EXPORT int customcpu_process(void *bmcpu_handle, int op_type, void *param,
                                  int param_size,
                                  const vector<float *> &input_tensors,
                                  const vector<vector<int>> &input_shapes,
                                  const vector<float *> &output_tensors,
                                  vector<vector<int>> &output_shapes);

int bmcpu_user_process(void *bmcpu_handle, void *param,
                       const vector<float *> &input_tensors,
                       const vector<vector<int>> &input_shapes,
                       const vector<float *> &output_tensors,
                       vector<vector<int>> &output_shapes);

/**
 * @name    bmcpu_reshape
 * @brief   output reshape with given input shape
 *
 * The interface will call change output shape with given input shape
 *
 * @param   [in]    bmcpu_handle   The pointer of cpu handler.
 * @param   [in]    op_type        The type of the cpu op that is defined in CPU_LAYER_TYPE.
 * @param   [in]    param          The pointer of the cpu op parameter.
 * @param   [in]    param_size     The byte size of the parameter.
 * @param   [in]    input_shapes   The shape of each input tensor.
 * @param   [in]    output_shapes  The shape of each output tensor.
 *
 * @retval  0      success
 * @retval  other  fail
 */
int  bmcpu_reshape(void* bmcpu_handle, int op_type,
                   void *param, int param_size,
                   const vector<vector<int>>& input_shapes,
                   vector<vector<int>>& output_shapes
                   );

int  bmcpu_user_reshape(void* bmcpu_handle, void *param,
                        const vector<vector<int>>& input_shapes,
                        vector<vector<int>>& output_shapes
                        );

/**
 * @name    bmcpu_dtype
 * @brief   get output dtypes with given input dtypes
 *
 * The interface will call to get output dtypes with given input dtypes
 *
 * @param   [in]    bmcpu_handle   The pointer of cpu handler.
 * @param   [in]    op_type        The type of the cpu op that is defined in CPU_LAYER_TYPE.
 * @param   [in]    input_dtypes   The dtype of each input tensor.
 * @param   [in]    output_dtypes  The dtype of each output tensor.
 *
 * @retval  0      success
 * @retval  other  fail
 */
int  bmcpu_dtype(void* bmcpu_handle, int op_type, const void *param,
                 size_t param_size,
                 const vector<int> &input_dtypes,
                 vector<int> &output_dtypes);
int  bmcpu_user_dtype(void* bmcpu_handle, void *param,
                      const vector<int> &input_dtypes,
                      vector<int> &output_dtypes);


#if defined (__cplusplus)
}
#endif


#endif /* _CPU_OP_H_ */
