package io.kinference.ndarray

class Strides(val shape: IntArray) {
    val strides = IntArray(shape.size)

    init {
        shape.foldRightIndexed(1) { index, i, acc ->
            strides[index] = acc
            acc * i
        }
    }


    fun offset(index: IntArray): Int {
        return index.foldIndexed(0) { ind, acc, i -> acc + i * strides[ind] }
    }

    fun index(offset: Int): IntArray {
        require(offset in 0 until linearSize) { "Offset $offset out of buffer bounds: (0, ${linearSize - 1})" }

        val res = IntArray(shape.size)
        var current = offset

        for ((index, stride) in strides.withIndex()) {
            res[index] = current / stride
            current %= stride
        }

        return res
    }

    fun transpose(permutations: IntArray): Strides {
        val newShape = IntArray(shape.size)
        for ((i, axis) in permutations.withIndex()) {
            newShape[i] = shape[axis]
        }

        return Strides(newShape)
    }

    val linearSize = if (shape.isEmpty()) 1 else strides[0] * shape[0]

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Strides) return false

        return shape.contentEquals(other.shape)
    }

    override fun hashCode() = shape.contentHashCode()

    companion object {
        val empty = Strides(IntArray(0))
    }
}
