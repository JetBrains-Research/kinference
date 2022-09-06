package io.kinference.ndarray

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

fun String.resolveTFJSDataType(): DataType {
    return when (this) {
        "float32" -> DataType.FLOAT
        "int32" -> DataType.INT
        "bool" -> DataType.BOOLEAN
        else -> error("Unsupported type: $this")
    }
}

inline fun <T> T.applyIf(predicate: Boolean, func: (T) -> (T)): T {
    return if (predicate) func(this) else this
}

fun makeNDArray(tfjsArray: ArrayTFJS, type: DataType): NDArrayTFJS {
    return when (type) {
        DataType.FLOAT, DataType.INT -> NumberNDArrayTFJS(tfjsArray)
        DataType.BOOLEAN -> BooleanNDArrayTFJS(tfjsArray)
        else -> error("Unsupported type: $type")
    }
}

fun makeNDArray(tfjsArray: ArrayTFJS, type: String) = makeNDArray(tfjsArray, type.resolveTFJSDataType())
