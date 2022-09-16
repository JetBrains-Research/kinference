package io.kinference.ndarray.arrays

class NDIndexIterator(
    val strides: Strides,
    from: IntArray = IntArray(strides.shape.size),
    to: IntArray = strides.shape.copyOf().onEach { it - 1 }
) : Iterator<IntArray> {
    private val indexSize = strides.shape.size
    private val maxElements = to.fold(1, Int::times) - from.fold(1, Int::times)
    private var elementsCounter = 0
    private var currentIndex = from.copyOf().apply { this[lastIndex] -= 1 }

    override fun hasNext(): Boolean = elementsCounter < maxElements

    override fun next(): IntArray {
        for (idx in indexSize - 1 downTo 0) {
            if (currentIndex[idx] != strides.shape[idx] - 1) {
                currentIndex[idx]++
                break
            }
            currentIndex[idx] = 0
        }
        elementsCounter++
        return currentIndex
    }
}

fun NDArray.ndIndexed(func: (IntArray) -> Unit) {
    return NDIndexIterator(strides).forEach(func)
}

fun NDArray.ndIndexed(from: IntArray, to: IntArray, func: (IntArray) -> Unit) {
    return NDIndexIterator(strides, from, to).forEach(func)
}
