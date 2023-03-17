package io.kinference.ndarray.math

import io.kinference.primitives.types.PrimitiveType

expect object FastMath {
    inline fun exp(value: Double): Double
    inline fun copySign(value: Double, sign: Double): Double
    inline fun copySign(value: Float, sign: Float): Float
    inline fun pow(value: Double, e: Long): Double
}

inline fun FastMath.exp(value: Float) = exp(value.toDouble()).toFloat()
inline fun FastMath.pow(value: Float, e: Long) = pow(value.toDouble(), e).toFloat()

inline fun FastMath.exp(value: PrimitiveType): PrimitiveType = error("")
inline fun FastMath.copySign(value: PrimitiveType, sign: PrimitiveType): PrimitiveType = error("")
inline fun FastMath.pow(value: PrimitiveType, e: Long): PrimitiveType = error("")
