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

internal fun computeColumnMajorIndex(
    inputInfo: InputInfo,
    dIter: IntArray,
    dOffset: IntArray,
    indexImR: Int
): Int {
    var indexImR1 = indexImR
    for (dI in inputInfo.rank - 1 downTo 0) {
        val dIm = dIter[dI] * inputInfo.strides[dI] - inputInfo.padBegin(dI) + dOffset[dI] * inputInfo.dilations[dI]

        indexImR1 *= inputInfo.inputShape[dI]
        indexImR1 += dIm
    }
    return indexImR1
}

internal fun isInPadding(actual: Int, bound: Int) : Boolean {
    return actual < 0 || actual >= bound
}
