package io.kinference.ndarray.math

import io.kinference.primitives.types.PrimitiveType

// This object provides functions from Apache Math 4 for JVM backend
// Exp function from Apache Math 4 significantly faster than default JVM realisation
// JS backend uses default realisations from stdlib
expect object FastMath {
    inline fun exp(value: Double): Double
    inline fun copySign(value: Double, sign: Double): Double
    inline fun copySign(value: Float, sign: Float): Float
}

inline fun FastMath.exp(value: Float) = exp(value.toDouble()).toFloat()

internal inline fun FastMath.exp(value: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
internal inline fun FastMath.copySign(value: PrimitiveType, sign: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
