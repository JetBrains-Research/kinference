package io.kinference.ndarray.extensions.utils

/***
 * Calculates the total size of the tensor with such shape.
 */
internal fun IntArray.inferShapeSize(): Int {
    return this.fold(1) { size, i -> size * i }
}

/***
 * Equivalent to ceil(a.toFloat() / b) but only integers are used.
 */
infix fun Int.divCeil(b: Int) = (this + b - 1) / b

/***
 * Amount of elements, ignoring first and second dimensions.
 */
internal fun calculateInnerShapeSize(shape: IntArray) = shape.foldIndexed(1) { index, acc, i -> if (index >= 2) acc * i else acc }

@Suppress("NAME_SHADOWING")
internal fun computeColumnMajorIndex(
    rowMajorIndex: Int,
    shape: IntArray
): Int {
    val index = IntArray(shape.size)
    var rowMajorIndex = rowMajorIndex
    for (i in shape.lastIndex downTo 0) {
        index[i] = rowMajorIndex % shape[i]
        rowMajorIndex /= shape[i]
    }
    var result = 0
    var shapeSize = 1
    for (i in 2 .. index.lastIndex) {
        result += shapeSize * index[i]
        shapeSize *= shape[i]
    }
    result += shapeSize * index[1]
    result += shapeSize * shape[1] * index[0]
    return result
}

internal fun isInPadding(actual: Int, bound: Int) : Boolean {
    return actual < 0 || actual >= bound
}