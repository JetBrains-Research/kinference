package io.kinference.trees

//TODO: SOFTMAX_ZERO, LOGISTIC, PROBIT in TFJS
enum class PostTransformType {
    NONE,
    SOFTMAX,
    SOFTMAX_ZERO,
    LOGISTIC,
    PROBIT
}
