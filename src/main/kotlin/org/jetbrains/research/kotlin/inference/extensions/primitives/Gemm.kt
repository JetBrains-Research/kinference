package org.jetbrains.research.kotlin.inference.extensions.primitives

import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray

fun gemm(m: Int, n: Int, k: Int, alpha: Float, a: FloatArray, lda: Int, b: FloatArray, ldb: Int,
         beta: Float, c: FloatArray, ldc: Int, aOffset: Int, bOffset: Int, cOffset: Int, transposeA: Boolean, transposeB: Boolean): FloatArray {
    if (beta != 1f) {
        for (i in 0 until m) {
            val cIdx = i * ldc + cOffset
            for (j in 0 until n) {
                c[cIdx + j] *= beta
            }
        }
    }

    when {
        transposeA && transposeB -> {
            for (t in 0 until m) {
                for (j in 0 until n) {
                    val cIdx = t * ldc + j + cOffset
                    for (i in 0 until k) {
                        val aIdx = i * lda + t + aOffset
                        val bIdx = j * ldb + i + bOffset
                        c[cIdx] += alpha * a[aIdx] * b[bIdx]
                    }
                }
            }
        }
        transposeA -> {
            for (t in 0 until m) {
                for (j in 0 until n) {
                    val cIdx = t * ldc + j + cOffset
                    for (i in 0 until k) {
                        val aIdx = i * lda + t + aOffset
                        val bIdx = i * ldb + j + bOffset
                        c[cIdx] += alpha * a[aIdx] * b[bIdx]
                    }
                }
            }
        }
        transposeB -> {
            for (t in 0 until m) {
                val aIdx = t * lda + aOffset
                for (j in 0 until n) {
                    val cIdx = t * ldc + j + cOffset
                    val bIdx = j * ldb + bOffset
                    for (i in 0 until k) {
                        c[cIdx] += alpha * a[aIdx + i] * b[bIdx + i]
                    }
                }
            }
        }
        else -> {
            for (t in 0 until m) {
                val cIdx = t * ldc + cOffset
                val aIdx = t * lda + aOffset
                for (i in 0 until k) {
                    val temp = alpha * a[aIdx + i]
                    val bIdx = i * ldb + bOffset
                    for (j in 0 until n) {
                        c[cIdx + j] += temp * b[bIdx + j]
                    }
                }
            }
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
