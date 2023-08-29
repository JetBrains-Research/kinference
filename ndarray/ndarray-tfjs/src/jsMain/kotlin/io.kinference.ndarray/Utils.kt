package io.kinference.ndarray

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.core.versionCpu
import io.kinference.ndarray.core.versionWebgl
import io.kinference.ndarray.extensions.unstack
import io.kinference.primitives.types.DataType

fun <T : NDArrayTFJS> Array<T>.update(i: Int, element: T) {
    this[i].close()
    this[i] = element
}

fun Array<NumberNDArrayTFJS>.unstackAs3DTypedArray(): Array<Array<MutableNumberNDArrayTFJS>> {
    return Array(this.size) { i ->
        val elementToArray = this[i].unstack()
        Array(elementToArray.size) { elementToArray[it].asMutable() }
    }
}

fun NumberNDArrayTFJS.unstackAs3DTypedArray(): Array<Array<MutableNumberNDArrayTFJS>> {
    val arrays = this.unstack()
    return arrays.unstackAs3DTypedArray()
}

fun Boolean.toInt() = if (this) 1 else 0

fun String.resolveTFJSDataType(): DataType {
    return when (this) {
        "float32" -> DataType.FLOAT
        "int32" -> DataType.INT
        "bool" -> DataType.BOOLEAN
        "string" -> DataType.ALL
        else -> error("Unsupported type: $this")
    }
}

internal fun DataType.resolveDTypeTFJS() = when (this) {
    DataType.DOUBLE, DataType.FLOAT -> "float32"
    DataType.BYTE, DataType.SHORT, DataType.INT, DataType.LONG,
    DataType.UBYTE, DataType.USHORT, DataType.UINT, DataType.ULONG -> "int32"
    DataType.BOOLEAN -> "bool"
    else -> error("Unsupported data type: $this")
}

inline fun <T> T.applyIf(predicate: Boolean, func: (T) -> (T)): T {
    return if (predicate) func(this) else this
}

internal fun makeNDArray(tfjsArray: ArrayTFJS, type: DataType): NDArrayTFJS {
    return when (type) {
        DataType.FLOAT, DataType.INT -> MutableNumberNDArrayTFJS(tfjsArray)
        DataType.BOOLEAN -> MutableBooleanNDArrayTFJS(tfjsArray)
        else -> error("Unsupported type: $type")
    }
}

internal fun makeNDArray(tfjsArray: ArrayTFJS, type: String) = makeNDArray(tfjsArray, type.resolveTFJSDataType())

internal fun activateCpuBackend() {
    versionCpu.length
}

internal fun activateWebglBackend() {
    versionWebgl.length
}

internal fun activateDefaultBackend() = activateWebglBackend()
