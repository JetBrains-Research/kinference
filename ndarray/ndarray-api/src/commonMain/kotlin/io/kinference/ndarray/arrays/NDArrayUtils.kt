package io.kinference.ndarray.arrays

fun IntProgression.toIntArray(): IntArray {
    val array = IntArray(this.count())
    for ((i, element) in this.withIndex()) {
        array[i] = element
    }
    return array
}

fun IntArray?.isNullOrEmpty() = this == null || this.isEmpty()
fun LongArray?.isNullOrEmpty() = this == null || this.isEmpty()

fun NDArray.isScalar() = shape.isEmpty()

fun NDArray.canDequantizePerAxis(axis: Int, zeroPoint: NDArray?, scale: NDArray): Boolean {
    return scale.rank == 1 && scale.linearSize == shape[axis] && (zeroPoint == null || zeroPoint.rank == 1 && zeroPoint.linearSize == shape[axis])
}

fun canDequantizePerTensor(zeroPoint: NDArray?, scale: NDArray): Boolean {
    return scale.linearSize == 1 && (zeroPoint == null || zeroPoint.linearSize == 1)
}

fun NDArray.indexAxis(axis: Int): Int {
    return if (axis < 0) rank + axis else axis
}

fun NDArray.computeBlockSize(fromDim: Int = 0, toDim: Int = this.shape.size): Int {
    return this.shape.sliceArray(fromDim until toDim).fold(1, Int::times)
}

fun NDArray.transpose(permutations: IntArray? = null): NDArray {
    require(permutations.isNullOrEmpty() || permutations!!.size == rank) { "Axes permutations list size should match the number of axes" }
    if (this.rank == 2) return this.transpose2D()

    val actualPerm = if (permutations.isNullOrEmpty()) shape.indices.reversed().toIntArray() else permutations
    return this.transpose(actualPerm!!)
}

fun broadcastShape(shapes: List<IntArray>): IntArray {
    val totalShapeLength = shapes.maxOf { it.size }

    return IntArray(totalShapeLength) { i ->
        val dims = shapes.map { it.getOrNull(it.size - i - 1) ?: 1 }
        val maxDim = dims.find { it != 1 } ?: 1

        if (dims.any { it != 1 && it != maxDim }) error("Cannot broadcast shapes")

        maxDim
    }.reversedArray()
}
