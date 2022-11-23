//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <float.h>

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

typedef enum reduce_type {
  REDUCE_MEAN = 0,
  REDUCE_SUM = 1,
  REDUCE_MAX = 2,
  REDUCE_MIN = 3,
  REDUCE_PROD = 4,
  REDUCE_ALL = 5,
  REDUCE_ANY = 6,
  REDUCE_L2  = 7,
  REDUCE_L1  = 8,
  REDUCE_SumSquare = 9,
  REDUCE_LogSum = 10,
  REDUCE_LogSumExp
} reduce_type_t;

static void fill(float *ptr,int size, float _v){
    for (int i = 0; i < size; i++)
        *ptr++ = _v;
}

template<typename T>
struct reduction_op_add
{
    T operator()(const T& x, const T& y) const
    {
        return x + y;
    }
};

template<typename T>
struct post_process_identity
{
    T operator()(const T& x) const
    {
        return x;
    }
};

template<typename T>
struct post_process_sqrt
{
    T operator()(const T& x) const
    {
        return static_cast<T>(sqrt(x));
    }
};

template<typename T>
struct post_process_log
{
    T operator()(const T& x) const
    {
        return static_cast<T>(log(x));
    }
};

template<typename T>
struct reduction_op_mul
{
    T operator()(const T& x, const T& y) const
    {
        return x * y;
    }
};

template<typename T>
struct reduction_op_asum
{
    T operator()(const T& x, const T& y) const
    {
        return static_cast<T>(x + fabs(y));
    }
};

template<typename T>
struct reduction_op_sumsq
{
    T operator()(const T& x, const T& y) const
    {
        return x + y * y;
    }
};

template<typename T>
struct reduction_op_sumsexp
{
    T operator()(const T& x, const T& y) const
    {
        return static_cast<T>(x + exp(y));
    }
};

template<typename T>
struct reduction_op_max
{
    T operator()(const T& x, const T& y) const
    {
        return std::max(x, y);
    }
};

template<typename T>
struct reduction_op_min
{
    T operator()(const T& x, const T& y) const
    {
        return std::min(x, y);
    }
};

template<typename Op, typename Op2>
static int reduction_op(float *a, float *b, std::vector<int> input_shape, int input_dims, float v0, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, int keepdims)
{
    Op op;
    Op2 op2;
    (void)(keepdims);

    if (input_dims == 1)
    {
        int w = input_shape[0];
        const float* ptr = static_cast<const float *>(a);

        float sum = v0;
        for (int i = 0; i < w; i++)
        {
            sum = op(sum, ptr[i]);
        }
        b[0] = sum;

        return 0;
    }

    if (input_dims == 2)
    {
        int w = input_shape[1];
        int h = input_shape[0];

        if (reduce_w && reduce_h)
        {
            float *sums = new float[h];
            #pragma omp parallel for num_threads(h)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = static_cast<const float *>(a) + i * w;

                float sum = v0;
                for (int j = 0; j < w; j++)
                {
                    sum = op(sum, ptr[j]);
                }
                sums[i] = sum;
            }

            float sum = v0;
            for (int i = 0; i < h; i++)
            {
                sum = op2(sum, sums[i]);
            }
            b[0] = sum;
            delete [] sums;
            return 0;
        }

        if (reduce_w && !reduce_h)
        {
            #pragma omp parallel for num_threads(h)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = static_cast<const float *>(a) + i * w;

                float sum = v0;
                for (int j = 0; j < w; j++)
                {
                    sum = op(sum, ptr[j]);
                }
                b[i] = sum;
            }
            return 0;
        }

        if (!reduce_w && reduce_h)
        {
            for (int i = 0; i < h; i++)
            {
                const float* ptr = static_cast<const float *>(a) + i * w;
                for (int j = 0; j < w; j++)
                {
                    b[j] = op(b[j], ptr[j]);
                }
            }
            return 0;
        }
    }

    if (input_dims == 3)
    {
        int w = input_shape[2];
        int h = input_shape[1];
        int channels = input_shape[0];
        int size = w * h;

        if (reduce_w && reduce_h && reduce_c)
        {
            float *sums = new float[channels];
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;

                float sum = v0;
                for (int i = 0; i < size; i++)
                {
                    sum = op(sum, ptr[i]);
                }
                sums[q] = sum;
            }

            float sum = v0;
            for (int i = 0; i < channels; i++)
            {
                sum = op2(sum, sums[i]);
            }
            b[0] = sum;
            delete []sums;
            return 0;
        }

        if (reduce_w && reduce_h && !reduce_c)
        {

            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;

                float sum = v0;
                for (int i = 0; i < size; i++)
                {
                    sum = op(sum, ptr[i]);
                }

                b[q] = sum;
            }

            return 0;
        }

        if (reduce_w && !reduce_h && reduce_c)
        {
            float *mins_ptr = new float[h];
            fill(mins_ptr, h, v0);
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                for (int i = 0; i < h; i++)
                {
                    float sum = v0;
                    for (int j = 0; j < w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    mins_ptr[i] = sum;
                    ptr += w;
                }
            }

            fill(b, h, v0);

            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    b[i] = op2(b[i], mins_ptr[i]);
                }
            }

            delete []mins_ptr;
            return 0;
        }

        if (!reduce_w && reduce_h && reduce_c)
        {
            float *mins = new float[w * channels];

            fill(mins, (w * channels), v0);
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float* mins_ptr = mins + q * w;

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        mins_ptr[j] = op(mins_ptr[j], ptr[j]);
                    }
                    ptr += w;
                }
            }

            memset(b, w, v0);
            for (int q = 0; q < channels; q++)
            {
                const float* mins_ptr = mins + q * w;
                for (int j = 0; j < w; j++)
                {
                    b[j] = op2(b[j], mins_ptr[j]);
                }
            }

            delete []mins;
            return 0;
        }

        if (reduce_w && !reduce_h && !reduce_c)
        {
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float* outptr = static_cast<float *>(b) + q * h;

                for (int i = 0; i < h; i++)
                {
                    float sum = v0;
                    for (int j = 0; j < w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    outptr[i] = sum;
                    ptr += w;
                }
            }

            return 0;
        }

        if (!reduce_w && !reduce_h && reduce_c)
        {
            fill(b, w * h , v0);
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;

                for (int i = 0; i < size; i++)
                {
                    b[i] = op(b[i], ptr[i]);
                }
            }

            return 0;
        }

        if (!reduce_w && reduce_h && !reduce_c)
        {
            fill(b, w * channels, v0);
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float* outptr = static_cast<float *>(b) + q * w;

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        outptr[j] = op(outptr[j], ptr[j]);
                    }
                    ptr += w;
                }
            }
            return 0;
        }
    }

    if (input_dims == 4)
    {
        int w = input_shape[3];
        int h = input_shape[2];
        int d = input_shape[1];
        int channels = input_shape[0];
        int size = w * h * d;

        if (reduce_w && reduce_h && reduce_d && reduce_c)
        {
            float *sums = new float[channels];

            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float sum = v0;
                for (int i = 0; i < size; i++)
                {
                    sum = op(sum, ptr[i]);
                }
                sums[q] = sum;
            }

            float sum = v0;
            for (int i = 0; i < channels; i++)
            {
                sum = op2(sum, sums[i]);
            }
            b[0] = sum;
            delete []sums;
            return 0;
        }

        if (reduce_w && reduce_h && reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;

                float sum = v0;
                for (int i = 0; i < size; i++)
                {
                    sum = op(sum, ptr[i]);
                }

                b[q] = sum;
            }

            return 0;
        }

        if (reduce_w && reduce_h && !reduce_d && reduce_c)
        {
            float *mins = new float[d * channels];
            fill(mins, d * channels, v0);

            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float* mins_ptr = mins + q * d;

                for (int i = 0; i < d; i++)
                {
                    float sum = v0;
                    for (int j = 0; j < w * h; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    mins_ptr[i] = sum;
                    ptr += w * h;
                }
            }

            fill(b, d, v0);
            for (int q = 0; q < channels; q++)
            {
                const float* mins_ptr = mins + q * d;
                for (int i = 0; i < d; i++)
                {
                    b[i] = op2(b[i], mins_ptr[i]);
                }
            }

            delete [] mins;
            return 0;
        }

        if (reduce_w && !reduce_h && reduce_d && reduce_c)
        {
            float *mins = new float[h * channels];
            fill(mins, h * channels, v0);

            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float* mins_ptr =  mins + q * h;

                for (int i = 0; i < d; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            mins_ptr[j] = op(mins_ptr[j], ptr[k]);
                        }
                        ptr += w;
                    }
                }
            }

            fill(b, h, v0);
            for (int q = 0; q < channels; q++)
            {
                const float* mins_ptr =mins + q * h;
                for (int i = 0; i < h; i++)
                {
                    b[i] = op2(b[i], mins_ptr[i]);
                }
            }

            delete []mins;
            return 0;
        }

        if (!reduce_w && reduce_h && reduce_d && reduce_c)
        {
            float *mins = new float[w * channels];
            fill(mins, w * channels, v0);

            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float* mins_ptr = mins + q * w;

                for (int i = 0; i < d; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            mins_ptr[k] = op(mins_ptr[k], ptr[k]);
                        }
                        ptr += w;
                    }
                }
            }

            fill(b, w, v0);
            for (int q = 0; q < channels; q++)
            {
                const float* mins_ptr = mins + q * w;
                for (int i = 0; i < w; i++)
                {
                    b[i] = op2(b[i], mins_ptr[i]);
                }
            }

            delete []mins;
            return 0;
        }

        if (reduce_w && reduce_h && !reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float* outptr = static_cast<float *>(b) + q * d;

                for (int i = 0; i < d; i++)
                {
                    float sum = v0;
                    for (int j = 0; j < w * h; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    outptr[i] = sum;
                    ptr += w * h;
                }
            }

            return 0;
        }

        if (reduce_w && !reduce_h && !reduce_d && reduce_c)
        {
            float *mins = new float[h*d*channels];
            fill(mins, (h*d*channels), v0);
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;

                float *minsm = mins + q * h * d;
                for (int i = 0; i < d; i++)
                {
                    float* mins_ptr = minsm + i * h;
                    for (int j = 0; j < h; j++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            mins_ptr[j] = op(mins_ptr[j], ptr[k]);
                        }
                        ptr += w;
                    }
                }
            }

            fill(b, h * d, v0);
            for (int q = 0; q < channels; q++)
            {
                float *minsm =  mins + q * h * d;
                for (int i = 0; i < d; i++)
                {
                    const float* mins_ptr =  minsm + i * h;
                    float* bptr = static_cast<float *>(b) + i * h;
                    for (int j = 0; j < h; j++)
                    {
                        bptr[j] = op2(bptr[j], mins_ptr[j]);
                    }
                }
            }
            delete []mins;
            return 0;
        }

        if (!reduce_w && !reduce_h && reduce_d && reduce_c)
        {

            float *mins = new float[w*h*channels];
            fill(mins, (w*h*channels), v0);
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float *minsm = mins + q * w * h;
                for (int i = 0; i < d; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        float* mins_ptr = minsm + j * w;
                        for (int k = 0; k < w; k++)
                        {
                            mins_ptr[k] = op(mins_ptr[k], ptr[k]);
                        }
                        ptr += w;
                    }
                }
            }

            fill(b, (w * h), v0);
            for (int q = 0; q < channels; q++)
            {
                float *minsm = mins + q * w * h;
                for (int i = 0; i < h; i++)
                {
                    float* mins_ptr = minsm + i * w;
                    float* bptr = b + i * w;
                    for (int j = 0; j < w; j++)
                    {
                        bptr[j] = op2(bptr[j], mins_ptr[j]);
                    }
                }
            }

            delete [] mins;
            return 0;
        }

        if (reduce_w && !reduce_h && reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float* outptr = static_cast<float *>(b) + q * h;
                for (int i = 0; i < h; i++)
                {
                    outptr[i] = v0;
                }

                for (int i = 0; i < d; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            outptr[j] = op(outptr[j], ptr[k]);
                        }
                        ptr += w;
                    }
                }
            }

            return 0;
        }

        if (!reduce_w && reduce_h && !reduce_d && reduce_c)
        {

            float *mins = new float[w*d*channels];
            fill(mins, (w*d*channels), v0);
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;

                float *minsm = mins + q * w * d;
                for (int i = 0; i < d; i++)
                {
                    float* mins_ptr =  minsm + i * w;
                    for (int j = 0; j < h; j++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            mins_ptr[k] = op(mins_ptr[k], ptr[k]);
                        }
                        ptr += w;
                    }
                }
            }

            fill(b, (w * d), v0);
            for (int q = 0; q < channels; q++)
            {
                float *minsm = mins + q * w * d;
                for (int i = 0; i < d; i++)
                {
                    const float* mins_ptr = minsm + i * w;
                    float* bptr = b + i * w;
                    for (int j = 0; j < w; j++)
                    {
                        bptr[j] = op2(bptr[j], mins_ptr[j]);
                    }
                }
            }

            delete []mins;
            return 0;
        }

        if (!reduce_w && reduce_h && reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float* outptr = static_cast<float *>(b) + q * w;
                for (int i = 0; i < w; i++)
                {
                    outptr[i] = v0;
                }

                for (int i = 0; i < d; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            outptr[k] = op(outptr[k], ptr[k]);
                        }
                        ptr += w;
                    }
                }
            }

            return 0;
        }

        if (reduce_w && !reduce_h && !reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;
                float* outptr = static_cast<float *>(b) + q * h * d;

                for (int i = 0; i < d * h; i++)
                {
                    float sum = v0;
                    for (int j = 0; j < w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    outptr[i] = sum;
                    ptr += w;
                }
            }

            return 0;
        }

        if (!reduce_w && !reduce_h && !reduce_d && reduce_c)
        {

            fill(b, (w * h * d), v0);
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;

                for (int i = 0; i < d; i++)
                {
                    float *outm = static_cast<float *>(b) + i * w * h;
                    for (int j = 0; j < h; j++)
                    {
                        float* outptr = outm + j * w;
                        for (int k = 0; k < w; k++)
                        {
                            outptr[k] = op(outptr[k], ptr[k]);
                        }
                        ptr += w;
                    }
                }
            }

            return 0;
        }

        if (!reduce_w && reduce_h && !reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;

                float *outm = b + q * w * d;
                fill(outm, (w * d), v0);

                for (int i = 0; i < d; i++)
                {
                    float* outptr = outm + i * w;
                    for (int j = 0; j < h; j++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            outptr[k] = op(outptr[k], ptr[k]);
                        }
                        ptr += w;
                    }
                }
            }

            return 0;
        }

        if (!reduce_w && !reduce_h && reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(channels)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = static_cast<const float *>(a) + q * size;

                float *outm  = b + q * w * h;
                fill(outm, (w * h), v0);
                for (int i = 0; i < d; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        float* outptr = outm + j * w;
                        for (int k = 0; k < w; k++)
                        {
                            outptr[k] = op(outptr[k], ptr[k]);
                        }
                        ptr += w;
                    }
                }
            }

            return 0;
        }
    }

    return 0;
}

template<typename MathOp>
static int reduction_post_process(float *a, std::vector<int> output_shape, int dims, float coeff)
{
    MathOp mathop;

    if (dims == 1)
    {
        int w = output_shape[0];

        #pragma omp parallel for num_threads(w)
        for (int i = 0; i < w; i++)
            a[i] = mathop(a[i]) * coeff;
    }
    else if (dims == 2)
    {
        int size = output_shape[0] * output_shape[1];

        #pragma omp parallel for num_threads(size)
        for (int i = 0; i < size; i++)
            a[i] = mathop(a[i]) * coeff;
    }
    else if (dims == 3)
    {
        int c = output_shape[0];
        int size = output_shape[1] * output_shape[2];
        if (c == 1)
        {
            #pragma omp parallel for num_threads(size)
            for (int i = 0; i < size; i++)
                a[i] = mathop(a[i]) * coeff;
        }
        else
        {
            #pragma omp parallel for num_threads(c)
            for (int q = 0; q < c; q++)
            {
                float* outptr = a + q * size;
                for (int i = 0; i < size; i++)
                    outptr[i] = mathop(outptr[i]) * coeff;
            }
        }
    } else if (dims == 4){
        int c = output_shape[0];
        int size = output_shape[1] * output_shape[2] * output_shape[3];
        if (c == 1)
        {
            #pragma omp parallel for num_threads(size)
            for (int i = 0; i < size; i++)
                a[i] = mathop(a[i]) * coeff;
        }
        else
        {
            #pragma omp parallel for num_threads(c)
            for (int q = 0; q < c; q++)
            {
                float* outptr = a + q * size;
                for (int i = 0; i < size; i++)
                    outptr[i] = mathop(outptr[i]) * coeff;
            }
        }
    }

    return 0;
}

template<typename Op, typename Op2, typename Op3>
static int reduce(float *a, float *b , std::vector<int> input_shape, int input_dims, std::vector<int> output_shape, int output_dims, float v0, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, bool post_process, float coeff, int keepdims)
{
    int ret = reduction_op<Op, Op2>(a, b, input_shape, input_dims, v0, reduce_w, reduce_h, reduce_d, reduce_c, keepdims);
    if (ret != 0)
        return -1;

    if (post_process || fabs(coeff - 1.f) > FLT_EPSILON)
    {
        ret = reduction_post_process<Op3>(b, output_shape, output_dims, coeff);
        if (ret != 0)
            return -1;
    }

    return 0;
}

LogicalResult tpu::ReduceOp::init(InferenceParameter &p) { return success(); }
void tpu::ReduceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ReduceOp::inference(InferenceParameter &p) {
    float *input_v = p.inputs[0];
    float *output_v = p.outputs[0];
    auto type_val = type();
    auto axes_val = Module::getI64Array(axes());
    auto keepdims_val = keepdims();
    auto out_shape = Module::getShape(output());
    auto input_shape = Module::getShape(input());
    // calc dims
    int num_dims = input_shape.size();
    int num_axes = axes_val->size();
    int output_dims = out_shape.size();
    float coeff_mean = 1.0f;

    std::vector<int> ishape(num_dims, 0);
    std::vector<int> oshape(output_dims, 0);
    for (int i =0; i < num_dims; i++)
        ishape[i] = input_shape[i];
    for(int i = 0; i< output_dims; i++)
        oshape[i] = out_shape[i];
    int is_reduce_orig[MAX_SHAPE_DIMS] = {0};
    int is_reduce[MAX_SHAPE_DIMS] = {0};
    int axis_list[MAX_SHAPE_DIMS] = {0};
    bool reduce_w = false;
    bool reduce_h = false;
    bool reduce_d = false;
    bool reduce_c = false;
    for (int i = 0; i < num_axes; i++){
        int axis = axes_val->at(i);
        if (axis < 0)
            axis += num_dims;
        is_reduce_orig[axis] = 1;
        axis_list[i] = axis;
    }

    //merge to 4 dims
    if (num_dims > 4) {
        int minimum_merged_dims = num_dims;
        int current_dims = 0;
        for (int i = 1; i < num_dims; ++i) {
            if (!is_reduce_orig[i-1] && !is_reduce_orig[i]) {
                minimum_merged_dims--;
            }
        }
        //if the minimum_merged_dims > 4, assert.
        assert(minimum_merged_dims <= 4);
        int pos = 0;
        int reduce_pos = 0;
        for (int i = 1; i < num_dims; ++i) {
            if (!is_reduce_orig[i-1] && !is_reduce_orig[i]
                && (num_dims - current_dims > 4)) {
                ishape[pos] *= ishape[i];
                current_dims++;
            } else {
                if (is_reduce_orig[i-1]) {
                    axis_list[reduce_pos++] = pos;
                }
                ishape[++pos] = ishape[i];
            }
        }
        if (is_reduce_orig[num_dims-1]) {
            axis_list[reduce_pos++] = pos;
        }
        ++pos;
        assert(pos == 4);
        num_dims = 4;
        ishape.resize(4);
        for (int i = 0; i < reduce_pos; i++){
            is_reduce[axis_list[i]] = 1;
        }
    } else {
        for (int i = 0; i < num_axes; i++) {
            is_reduce[axis_list[i]] = 1;
        }
    }

    if (num_dims == 1){
        reduce_w = true;
    } else if (num_dims == 2){
        if (is_reduce[0] == 1) reduce_h = true;
        if (is_reduce[1] == 1) reduce_w = true;
    } else if (num_dims == 3){
        if (is_reduce[0] == 1) reduce_c = true;
        if (is_reduce[1] == 1) reduce_h = true;
        if (is_reduce[2] == 1) reduce_w = true;
    } else if (num_dims == 4){
        if (is_reduce[0] == 1) reduce_c = true;
        if (is_reduce[1] == 1) reduce_d = true;
        if (is_reduce[2] == 1) reduce_h = true;
        if (is_reduce[3] == 1) reduce_w = true;
    }

    if (type_val == REDUCE_SUM)
        reduce<reduction_op_add<float>, reduction_op_add<float>, post_process_identity<float> >(input_v, output_v,
                                              ishape, num_dims, oshape, output_dims, 0.f, reduce_w, reduce_h, reduce_d, reduce_c, false, coeff_mean, static_cast<int>(keepdims_val));
    if (type_val == REDUCE_SumSquare)
        reduce<reduction_op_sumsq<float>, reduction_op_add<float>, post_process_identity<float> >(input_v, output_v,
                                              ishape, num_dims, oshape, output_dims, 0.f, reduce_w, reduce_h, reduce_d, reduce_c, false, coeff_mean, static_cast<int>(keepdims_val));
    if (type_val == REDUCE_MEAN)
    {
        int scale = 1;
        if (num_dims == 1)
        {
            scale = ishape[0];
        }
        else if (num_dims == 2)
        {
            if (reduce_w) scale *= ishape[1];
            if (reduce_h) scale *= ishape[0];
        }
        else if (num_dims == 3)
        {
            if (reduce_w) scale *= ishape[2];
            if (reduce_h) scale *= ishape[1];
            if (reduce_c) scale *= ishape[0];
        }
        else if (num_dims == 4)
        {
            if (reduce_w) scale *= ishape[3];
            if (reduce_h) scale *= ishape[2];
            if (reduce_d) scale *= ishape[1];
            if (reduce_c) scale *= ishape[0];
        }

        coeff_mean = 1.0f / scale;
        reduce<reduction_op_add<float>, reduction_op_add<float>, post_process_identity<float> >(input_v, output_v,
                                              ishape, num_dims, oshape, output_dims, 0.f, reduce_w, reduce_h, reduce_d, reduce_c, true, coeff_mean, static_cast<int>(keepdims_val));
    }

    if (type_val == REDUCE_MAX)
        reduce<reduction_op_max<float>, reduction_op_max<float>, post_process_identity<float> >(input_v, output_v,
                                              ishape, num_dims, oshape, output_dims, -FLT_MAX, reduce_w, reduce_h, reduce_d, reduce_c, false, coeff_mean, static_cast<int>(keepdims_val));

    if (type_val == REDUCE_MIN)
        reduce<reduction_op_min<float>, reduction_op_min<float>, post_process_identity<float> >(input_v, output_v,
                                              ishape, num_dims, oshape, output_dims, FLT_MAX, reduce_w, reduce_h, reduce_d, reduce_c, false, coeff_mean, static_cast<int>(keepdims_val));

    if (type_val == REDUCE_PROD)
        reduce<reduction_op_mul<float>, reduction_op_mul<float>, post_process_identity<float> >(input_v, output_v,
                                              ishape, num_dims, oshape, output_dims, 1.f, reduce_w, reduce_h, reduce_d, reduce_c, false, coeff_mean, static_cast<int>(keepdims_val));

    if (type_val == REDUCE_L1)
        reduce<reduction_op_asum<float>, reduction_op_add<float>, post_process_identity<float> >(input_v, output_v,
                                              ishape, num_dims, oshape, output_dims, 0.f, reduce_w, reduce_h, reduce_d, reduce_c, false, 1.f, static_cast<int>(keepdims_val));

    if (type_val == REDUCE_L2)
        reduce<reduction_op_sumsq<float>, reduction_op_add<float>, post_process_sqrt<float> >(input_v, output_v,
                                              ishape, num_dims, oshape, output_dims, 0.f, reduce_w, reduce_h, reduce_d, reduce_c, true, 1.f, static_cast<int>(keepdims_val));

    if (type_val == REDUCE_LogSum)
        reduce<reduction_op_add<float>, reduction_op_add<float>, post_process_log<float> >(input_v, output_v,
                                              ishape, num_dims, oshape, output_dims, 0.f, reduce_w, reduce_h, reduce_d, reduce_c, true, 1.f, static_cast<int>(keepdims_val));

    if (type_val == REDUCE_LogSumExp)
        reduce<reduction_op_sumsexp<float>, reduction_op_add<float>, post_process_log<float> >(input_v, output_v,
                                              ishape, num_dims, oshape, output_dims, 0.f, reduce_w, reduce_h, reduce_d, reduce_c, true, 1.f, static_cast<int>(keepdims_val));

    return success();
}
