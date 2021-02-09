package io.kinference.ndarray.extensions

import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.concat

fun gemm(m: Int, n: Int, k: Int, alpha: Double, a: NumberNDArray, b: NumberNDArray, beta: Double, c: MutableNDArray,
         aOffset: Int = 0, bOffset: Int = 0, cOffset: Int = 0, transposeA: Boolean = false, transposeB: Boolean = false) : MutableNDArray {
    val lda = if (transposeA) m else k
    val ldb = if (transposeB) k else n
    return a.gemm(m, n, k, alpha, lda, b, ldb, beta, c, n, aOffset, bOffset, cOffset, transposeA, transposeB)
}

infix fun NumberNDArray.matmul(other: NumberNDArray): MutableNumberNDArray {
    val outputShape = Broadcasting.broadcastShapeForMatmul(this.shape, other.shape)
    val outputArray = allocateNDArray(Strides(outputShape))
    return matmul(other, outputArray) { otherArray, dest -> this.dot(otherArray, dest) }
}

private fun NumberNDArray.matmul(other: NumberNDArray, dest: MutableNumberNDArray,
                                 dotFunc: NumberNDArray.(NumberNDArray, MutableNumberNDArray) -> MutableNumberNDArray
): MutableNumberNDArray {
    require(!this.isScalar() && !other.isScalar()) { "Matmul operation is not available for scalar tensors" }

    if (rank <= 2 && other.rank <= 2) {
        val actualThis = if (rank == 1) this.reshapeView(1.concat(shape)) as NumberNDArray else this
        val actualOther = if (other.rank == 1) this.reshapeView(other.shape.concat(1)) else other

        return actualThis.dotFunc(actualOther as NumberNDArray, dest)
    }

    Broadcasting.matmulWithBroadcast(this, other, dest, dotFunc)
    return dest
}
