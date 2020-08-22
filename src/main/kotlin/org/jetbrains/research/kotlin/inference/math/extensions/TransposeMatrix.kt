package org.jetbrains.research.kotlin.inference.math.extensions

/*import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun transpose(array: FloatArray, rowNum: Int, colNum: Int): FloatArray {
    val tmp = array.copyOf()

    for (j in (0 until colNum)) {
        val ind = j * rowNum
        for (i in (0 until rowNum)) {
            array[ind + i] = tmp[i * colNum + j]
        }
    }

    return array
}

fun transpose(array: DoubleArray, rowNum: Int, colNum: Int): DoubleArray {
    val tmp = array.copyOf()

    for (j in (0 until colNum)) {
        val ind = j * rowNum
        for (i in (0 until rowNum)) {
            array[ind + i] = tmp[i * colNum + j]
        }
    }

    return array
}

fun transpose(array: IntArray, rowNum: Int, colNum: Int): IntArray {
    val tmp = array.copyOf()

    for (j in (0 until colNum)) {
        val ind = j * rowNum
        for (i in (0 until rowNum)) {
            array[ind + i] = tmp[i * colNum + j]
        }
    }

    return array
}

fun transpose(array: LongArray, rowNum: Int, colNum: Int): LongArray {
    val tmp = array.copyOf()

    for (j in (0 until colNum)) {
        val ind = j * rowNum
        for (i in (0 until rowNum)) {
            array[ind + i] = tmp[i * colNum + j]
        }
    }

    return array
}

fun transpose(array: ShortArray, rowNum: Int, colNum: Int): ShortArray {
    val tmp = array.copyOf()

    for (j in (0 until colNum)) {
        val ind = j * rowNum
        for (i in (0 until rowNum)) {
            array[ind + i] = tmp[i * colNum + j]
        }
    }

    return array
}

fun <T> MutableTypedNDArray<T>.matrixTranspose(): MutableTypedNDArray<T> {
    require(this.shape.size == 2)
    val newShape = shape.reversedArray()
    val newStrides = Strides(newShape)

    when (array) {
        is IntArray -> transpose(array as IntArray, shape[0], shape[1])
        is FloatArray -> transpose(array as FloatArray, shape[0], shape[1])
        is ShortArray -> transpose(array as ShortArray, shape[0], shape[1])
        is DoubleArray -> transpose(array as DoubleArray, shape[0], shape[1])
        is LongArray -> transpose(array as LongArray, shape[0], shape[1])
        else -> throw UnsupportedOperationException()
    }

    return this.reshape(newStrides)
}*/
