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

inline fun fgelu(x: Float): Float {
    return x * (0.5f + 0.5f * tanh(x * (0.035677408136300125f * x * x + 0.7978845608028654f)))
}

inline fun fgelu(x: Double): Double {
    return x * (0.5 + 0.5 * tanh(x * (0.035677408136300125 * x * x + 0.7978845608028654)))
}
