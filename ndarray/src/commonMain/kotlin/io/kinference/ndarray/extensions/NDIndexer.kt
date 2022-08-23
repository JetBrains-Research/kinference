package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.NDArray

class NDIndexIterator(val shape: IntArray) : Iterator<IntArray> {
    private val indexSize = shape.size
    private val maxElements = shape.fold(1, Int::times)
    private var elementsCounter = 0
    private var currentIndex = IntArray(indexSize).apply { this[lastIndex] = -1 }

    override fun hasNext(): Boolean = elementsCounter < maxElements

    override fun next(): IntArray {
        for (idx in indexSize - 1 downTo 0) {
            if (currentIndex[idx] != shape[idx] - 1) {
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
    return NDIndexIterator(this.shape).forEach(func)
}
