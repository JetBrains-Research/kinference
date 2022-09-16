package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import io.kinference.utils.runBlocking
import kotlinx.coroutines.launch
import kotlin.coroutines.CoroutineContext
import kotlin.math.min

private fun expFunc(type: DataType) = when (type) {
    DataType.FLOAT -> object : FloatMap {
        override fun apply(value: Float): Float = kotlin.math.exp(value)
    }

    DataType.DOUBLE -> object : DoubleMap {
        override fun apply(value: Double): Double = kotlin.math.exp(value)
    }
    else -> error("Unsupported data type: $type")
}

private fun MutableNumberNDArrayCore.softmaxRow() {
    minusAssign(createScalarNDArray(type, max()) as NumberNDArray)
    mapMutable(expFunc(type))
    divAssign(createScalarNDArray(type, sum()) as NumberNDArray)
}

fun softmax(
    input: NDArrayCore,
    axis: Int = 0,
    strides: Strides = input.strides,
    coroutineContext: CoroutineContext? = null
): MutableNumberNDArrayCore {
    fun resolveDims(dims: IntArray?): Int {
        return if (dims.isNullOrEmpty()) 1 else dims!!.reduce(Int::times)
    }

    val actualAxis = input.indexAxis(axis)
    val shape = input.shape
    val (rowIdx, columnIdx) = (shape.indices).partition { it < actualAxis }

    val rows = resolveDims(shape.sliceArray(rowIdx))
    val columns = resolveDims(shape.sliceArray(columnIdx))

    val matrix = input.reshape(intArrayOf(rows, columns))
    val matrixRows = Array(rows) { matrix.row(it) as MutableNumberNDArrayCore }

    if (matrixRows.size > 128 && coroutineContext != null) {
        runBlocking(coroutineContext) {
            for (i in matrixRows.indices step 32) {
                val end = min(i + 32, matrixRows.size)
                launch {
                    for (row in i until end) {
                        matrixRows[row].softmaxRow()
                    }
                }
            }
        }
    } else {
        for (i in matrixRows.indices) {
            matrixRows[i].softmaxRow()
        }
    }

    val step = matrixRows[0].linearSize
    val array = allocateNDArray(input.type, strides) as MutableNumberNDArrayCore
    repeat(matrixRows.size) { i ->
        array.copyFrom(i * step, matrixRows[i])
    }
    return array
}
