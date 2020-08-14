package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray

fun gemm(m: Int, n: Int, k: Int, alpha: Float, a: FloatArray, lda: Int, b: FloatArray, ldb: Int,
         beta: Float, c: FloatArray, ldc: Int, aOffset: Int, bOffset: Int, cOffset: Int, transposeA: Boolean, transposeB: Boolean): FloatArray {
    if (beta != 1f) {
        for (i in 0 until m) {
            for (j in 0 until n) {
                val idxCount = i * ldc + j + cOffset
                c[idxCount] *= beta
            }
        }
    }
    for (t in 0 until m) {
        for (j in 0 until n) {
            var sum = 0f
            for (i in 0 until k) {
                val aIdx = if (!transposeA) t * lda + i + aOffset else i * lda + t + aOffset
                val bIdx = if (!transposeB) i * ldb + j + bOffset else j * ldb + i + bOffset
                sum += alpha * a[aIdx] * b[bIdx]
            }
            c[t * ldc + j + cOffset] += sum
        }
    }
    return c
}

fun gemm(a: TypedNDArray<Any>, b: TypedNDArray<Any>, c: TypedNDArray<Any>, aOffset: Int, bOffset: Int, cOffset: Int, m: Int, n: Int, k: Int, lda: Int, ldb: Int, ldc: Int, alpha: Double = 1.0, beta: Double = 1.0, transposeA: Boolean = false, transposeB: Boolean = false): TypedNDArray<Any> {
    when (a.array) {
        is FloatArray -> gemm(m, n, k, alpha.toFloat(), a.array as FloatArray, lda, b.array as FloatArray, ldb, beta.toFloat(), c.array as FloatArray, ldc, aOffset, bOffset, cOffset, transposeA, transposeB)
        else -> throw UnsupportedOperationException()
    }

    return c
}
