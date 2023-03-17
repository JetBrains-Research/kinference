package io.kinference.ndarray.math

import io.kinference.primitives.types.PrimitiveType

expect object FastMath {
    inline fun exp(value: Double): Double
    inline fun copySign(value: Double, sign: Double): Double
    inline fun copySign(value: Float, sign: Float): Float
}

inline fun FastMath.exp(value: Float) = exp(value.toDouble()).toFloat()

inline fun FastMath.exp(value: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
inline fun FastMath.copySign(value: PrimitiveType, sign: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
