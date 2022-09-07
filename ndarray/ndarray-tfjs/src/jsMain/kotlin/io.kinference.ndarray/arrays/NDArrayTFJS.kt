package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.utils.Closeable
import io.kinference.primitives.types.DataType

abstract class NDArrayTFJS(tfjsArray: ArrayTFJS) : NDArray {
    var tfjsArray = tfjsArray
        protected set

    override val strides
        get() = Strides(tfjsArray.shape.toIntArray())

    override val type: DataType = tfjsArray.dtype.resolveTFJSDataType()

    override fun close() {
        tfjsArray.dispose()
    }
}

val NDArrayTFJS.dtype: String
    get() = tfjsArray.dtype

val NDArrayTFJS.shapeArray: Array<Int>
    get() = tfjsArray.shape
