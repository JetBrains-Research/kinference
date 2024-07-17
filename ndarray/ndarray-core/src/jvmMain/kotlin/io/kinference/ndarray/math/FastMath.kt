package io.kinference.ndarray.math

import io.kinference.primitives.types.PrimitiveType
import org.apache.commons.math4.core.jdkmath.AccurateMath

// This object provides functions from Apache Math 4 for JVM backend
// Exp function from Apache Math 4 significantly faster than default JVM realisation
object FastMath {
    inline fun exp(value: Double): Double = AccurateMath.exp(value)
    inline fun copySign(value: Double, sign: Double): Double = AccurateMath.copySign(value, sign)
    inline fun copySign(value: Float, sign: Float): Float = AccurateMath.copySign(value, sign)

    internal inline fun exp(value: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
    internal inline fun copySign(value: PrimitiveType, sign: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
}

inline fun FastMath.exp(value: Float) = exp(value.toDouble()).toFloat()
inline fun FastMath.exp(value: Int) = exp(value.toDouble()).toInt()
inline fun FastMath.exp(value: Long) = exp(value.toDouble()).toLong()
inline fun FastMath.exp(value: UInt) = exp(value.toDouble()).toUInt()
inline fun FastMath.exp(value: ULong) = exp(value.toDouble()).toULong()

