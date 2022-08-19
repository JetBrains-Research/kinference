package io.kinference.ndarray.extensions

import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

inline fun <reified T> createTiledArray(type: DataType, vararg shape: Int, noinline init: (Int) -> T): Any {
    return createTiledArray(type, shape, init)
}

fun tiledFromPrimitiveArray(array: Any, vararg shape: Int): Any {
    return tiledFromPrimitiveArray(shape, array)
}


fun createMutableNDArray(type: DataType, value: Any, vararg shape: Int): MutableNDArray {
    return createMutableNDArray(type, value, Strides(shape))
}

fun createNDArray(type: DataType, value: Any, vararg shape: Int): NDArray {
    return createNDArray(type, value, Strides(shape))
}

fun allocateNDArray(type: DataType, vararg shape: Int) = allocateNDArray(type, Strides(shape))
