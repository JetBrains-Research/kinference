package io.kinference.core.operators.math


import kotlin.math.exp

inline fun tanh(value: Float): Float {
    var temp = exp(2.0f * value)
    if (temp.isInfinite()) temp = Float.MAX_VALUE
    return ((temp - 1.0f) / (temp + 1.0f))
}

inline fun tanh(value: Double): Double {
    return ((exp(2.0 * value) - 1.0) / (exp(2.0 * value) + 1.0))
}
