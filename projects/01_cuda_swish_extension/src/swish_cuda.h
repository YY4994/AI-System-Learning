#ifndef SWISH_H
#define SWISH_H

#include <cmath>
#include <cstdint>
#include <vector>

namespace swish
{
    class SwishFunction
    {
    public:
        // 静态方法，便于调用
        static void forward_cpu(const float *x, float *y, float *saved_s, int64_t n);
        static void backward_cpu(const float *grad_output, const float *x, const float *saved_s, float *grad_input, int64_t n);

        // (可选) 也可以作为实例类，保存中间状态
    };
}

#endif // SWISH_H