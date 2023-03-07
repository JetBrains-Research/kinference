package io.kinference.ndarray.extensions.utils

/***
 * Equivalent to ceil(a.toFloat() / b) but only integers are used.
 */
infix fun Int.divCeil(b: Int) = (this + b - 1) / b

/***
 * Extends shape of tensor with given pads. Ignores 1st and 2nd dimension.
 */
internal fun getShapeWithPads(shape: IntArray, pads: IntArray) =
    IntArray(shape.size) { i -> if (i < 2) shape[i] else shape[i] + pads[i - 2] + pads[i + shape.size - 4] }

/***
 * Extends shape of tensor with given dilations. Ignores 1st and 2nd dimension.
 */
internal fun getShapeWithDilations(shape: IntArray, dilations: IntArray) =
    IntArray(shape.size) { i -> if (i < 2) shape[i] else (shape[i] - 1) * dilations[i - 2] + 1 }

internal operator fun IntArray.times(oth: IntArray) = IntArray(this.size) { if (it < 2) 0 else this[it] * oth[it - 2] }

/***
 * Calculates raw shift of index, ignoring first and second dimensions.
 */
internal fun calculateInnerShift(shape: IntArray, index: IntArray): Int {
    var shift = 0
    var cur = 1
    for (j in shape.lastIndex downTo 2) {
        shift += index[j] * cur
        cur *= shape[j]
    }
    return shift
}

/***
 * Amount of elements, ignoring first and second dimensions.
 */
internal fun calculateInnerShapeSize(shape: IntArray) = shape.foldIndexed(1) { index, acc, i -> if (index >= 2) acc * i else acc }

/***
 * Calculates the raw position of the following type of indexes [pos0, pos1, 0, 0, ..., 0].
 */
internal fun calculateSignificantShift(pos0: Int, pos1: Int, shape1: Int, innerShapeSize: Int) = ((pos0 * shape1) + pos1) * innerShapeSize

internal operator fun IntArray.plus(oth: IntArray) = IntArray(this.size) { this[it] + oth[it] }

internal operator fun IntArray.minus(oth: IntArray) = IntArray(this.size) { this[it] - oth[it] }

/***
 * Iterates over elements inside IntProgression.
 */
internal class ProgressionIterator(
    internal val progression: IntProgression
) : Iterator<Int> {
    internal var cur = progression.first - progression.step
    override fun hasNext(): Boolean {
        return cur + progression.step <= progression.last
    }

    override fun next(): Int {
        cur += progression.step
        return cur
    }

    fun reset() {
        cur = progression.first - progression.step
    }

    fun decrement() {
        cur -= progression.step
    }
}
