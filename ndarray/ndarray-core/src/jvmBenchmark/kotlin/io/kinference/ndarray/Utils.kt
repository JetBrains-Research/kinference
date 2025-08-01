package io.kinference.ndarray

fun randomFloat(l: Float = -1024f, r: Float = 1024f) =
    (r - l) * Math.random().toFloat() + l

fun randomDouble(l: Double = -1024.0, r: Double = 1024.0) =
    (r - l) * Math.random() + l
