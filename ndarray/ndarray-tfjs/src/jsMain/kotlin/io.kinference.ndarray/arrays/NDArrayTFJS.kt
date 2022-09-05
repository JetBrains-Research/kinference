package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.utils.Closeable
import io.kinference.primitives.types.DataType

abstract class NDArrayTFJS(tfjsArray: ArrayTFJS) : NDArray, Closeable {
    var tfjsArray = tfjsArray
        protected set

    override val strides
        get() = Strides(tfjsArray.shape.toIntArray())

    override val type: DataType = tfjsArray.dtype.resolveDataType()

    override fun close() {
        tfjsArray.dispose()
    }
}

val NDArrayTFJS.dtype: String
    get() = tfjsArray.dtype

val NDArrayTFJS.shapeArray: Array<Int>
    get() = tfjsArray.shape

fun <T : Closeable> closeAll(arrays: Array<T>) = arrays.forEach { it.close() }
fun <T : Closeable> closeAll(arrays: List<T>) = arrays.forEach { it.close() }
fun <T : Closeable> closeAll(vararg array: T?) = array.forEach { it?.close() }
