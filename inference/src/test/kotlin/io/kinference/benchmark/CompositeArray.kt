package io.kinference.benchmark

import io.kinference.ndarray.Strides

class CompositeArray(val strides: Strides, init: () -> Float) {
    val size = strides.linearSize
    val blockSize = strides.shape.last()
    val blocksNum = size / blockSize
    val blocks = Array(blocksNum) { FloatArray(blockSize) { init() } }
}
