package io.kinference.webgpu.ndarray

import io.kinference.ndarray.Strides

class ArrayInfo(
    val strides: Strides,
    val type: WebGPUDataType
) {
    val shape: IntArray
        get() = strides.shape
    val size
        get() = strides.linearSize
    val sizeBytes
        get() = size * type.sizeBytes
    val rank: Int
        get() = shape.size

    constructor(shape: IntArray, type: WebGPUDataType) : this(Strides(shape), type)
}
