package io.kinference.ndarray

import io.kinference.ndarray.extensions.constants.DoubleConstants
import io.kinference.ndarray.math.FastMath
import jdk.incubator.vector.*
import kotlin.math.abs

fun logistic(inputBlock: DoubleArray, outputBlock: DoubleArray) {
    val inputBlockSize = inputBlock.size
    val _vecLen_0 = DoubleVector.SPECIES_PREFERRED.length()
    val _vecEnd_0 = inputBlockSize - (inputBlockSize % _vecLen_0)
    val one = 1.0
    val zero = 0.0

    for (_vec_internal_idx in 0 until _vecEnd_0 step _vecLen_0) {
        val input_vec = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, inputBlock, 0 + _vec_internal_idx)
        val mid_vec = DoubleVector.broadcast(DoubleVector.SPECIES_PREFERRED, one)
            .lanewise(
                VectorOperators.DIV, DoubleVector.broadcast(DoubleVector.SPECIES_PREFERRED, DoubleConstants.ONE + DoubleConstants.ZERO)
                    .lanewise(
                        VectorOperators.ADD, input_vec
                            .lanewise(VectorOperators.ABS)
                            .lanewise(VectorOperators.NEG)
                            .lanewise(VectorOperators.EXP)
                    )
            )
        DoubleVector.broadcast(DoubleVector.SPECIES_PREFERRED, one)
            .lanewise(VectorOperators.SUB, mid_vec).blend(
                mid_vec, input_vec
                    .compare(VectorOperators.GE, DoubleVector.broadcast(DoubleVector.SPECIES_PREFERRED, zero))
            )
            .intoArray(outputBlock, 0 + _vec_internal_idx)
    }
    for (_vec_internal_idx in _vecEnd_0 until inputBlockSize) {
        val input_lin = inputBlock[0 + _vec_internal_idx]
        val mid_lin = (one / (DoubleConstants.ONE + DoubleConstants.ZERO + FastMath.exp((-abs(input_lin)))))
        val a_lin = (if ((input_lin >= zero)) mid_lin else (one - mid_lin))

        outputBlock[0 + _vec_internal_idx] = a_lin.toDouble()
    }
}

fun main() {
    val l = 10000
    val a = DoubleArray(l) { randomDouble(-10000.0, 10000.0) }
    val b = DoubleArray(l)
    var c = 0.0
    repeat(2) {
        for (i in 0 until l) a[i] = b[i]
        logistic(a, b)
        c += b[10]
    }
    println()
}
