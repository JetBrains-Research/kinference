package io.kinference.ndarray.arrays

class NDIndexer(
    val strides: Strides,
    from: IntArray = strides.defaultStart(),
    to: IntArray = strides.defaultEnd()
) : Iterator<IntArray> {
    constructor(shape: IntArray) : this(Strides(shape))
    private val indexSize = strides.shape.size
    private val maxElements = strides.countElements(from, to)
    private val currentIndex = from.copyOf().apply { this[lastIndex] -= 1 }

    var linearIndex = 0
        private set

    override fun hasNext(): Boolean = linearIndex < maxElements

    override fun next(): IntArray {
        for (idx in indexSize - 1 downTo 0) {
            if (currentIndex[idx] != strides.shape[idx] - 1) {
                currentIndex[idx]++
                break
            }
            currentIndex[idx] = 0
        }
        linearIndex++
        return currentIndex
    }

    companion object {
        private fun Strides.countElements(from: IntArray, to: IntArray) = offset(to) - offset(from) + 1
    }
}

private fun Strides.defaultStart() = IntArray(shape.size)
private fun Strides.defaultEnd() = IntArray(shape.size) { shape[it] - 1 }

fun <T : NDArray> T.ndIndices(
    from: IntArray = strides.defaultStart(),
    to: IntArray = strides.defaultEnd(),
    func: (IntArray) -> Unit
) = NDIndexer(strides, from, to).forEach(func)
