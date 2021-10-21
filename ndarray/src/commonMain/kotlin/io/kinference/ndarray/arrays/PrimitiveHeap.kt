@file:GeneratePrimitives(DataType.NUMBER)
@file:Suppress("DuplicatedCode")

package io.kinference.ndarray.arrays

import io.kinference.ndarray.reversed
import io.kinference.ndarray.swap
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*

@GenerateNameFromPrimitives
internal class PrimitiveMaxHeap(size: Int) {
    val data = PrimitiveArray(size)
    val indices = IntArray(size)

    private var count = 0

    val minValue: PrimitiveType
        get() = data[0]

    private val minIndex: Int
        get() = indices[0]

    private fun siftUp(idx: Int) {
        var internalIdx = idx
        while (data[idx] < data[(idx - 1) / 2]) {
            data.swap(idx, (idx - 1) / 2)
            indices.swap(idx, (idx - 1) / 2)
            internalIdx = (internalIdx - 1) / 2
        }
    }

    private fun siftDown(idx: Int) {
        var internalIdx = idx
        while (2 * internalIdx + 1 < count) {
            val left = 2 * internalIdx + 1
            val right = left + 1

            val j = if (right < count && data[right] < data[left]) right else left

            if (data[internalIdx] <= data[j])
                break

            data.swap(internalIdx, j)
            indices.swap(internalIdx, j)
            internalIdx = j
        }
    }

    fun insert(key: PrimitiveType, index: Int) {
        count++
        indices[count - 1] = index
        data[count - 1] = key
        siftUp(count - 1)
    }

    fun removeMin() {
        data[0] = data[count - 1]
        indices[0] = indices[count - 1]
        count--
        siftDown(0)
    }

    fun sorted(): Pair<PrimitiveArray, IntArray> {
        val sortedData = PrimitiveArray(count)
        val sortedIndices = IntArray(count)

        for (idx in (count - 1) downTo 0) {
            sortedData[idx] = minValue
            sortedIndices[idx] = minIndex

            removeMin()
        }

        return sortedData to sortedIndices
    }

    fun clear() {
        count = 0
    }
}

@GenerateNameFromPrimitives
internal class PrimitiveMinHeap(size: Int) {
    val data = PrimitiveArray(size)
    val indices = IntArray(size)

    private var count = 0

    val maxValue: PrimitiveType
        get() = data[0]

    private val maxIndex: Int
        get() = indices[0]

    private fun siftUp(idx: Int) {
        var internalIdx = idx
        while (data[idx] > data[(idx - 1) / 2]) {
            indices.swap(idx, (idx - 1) / 2)
            data.swap(idx, (idx - 1) / 2)
            internalIdx = (internalIdx - 1) /2
        }
    }

    private fun siftDown(idx: Int) {
        var internalIdx = idx
        while (2 * internalIdx + 1 < count) {
            val left = 2 * internalIdx + 1
            val right = left + 1

            val j = if (right < count && data[right] > data[left]) right else left

            if (data[internalIdx] >= data[j])
                break

            data.swap(internalIdx, j)
            indices.swap(internalIdx, j)
            internalIdx = j
        }
    }

    fun insert(key: PrimitiveType, index: Int) {
        count++
        data[count - 1] = key
        indices[count - 1] = index
        siftUp(count - 1)
    }

    fun removeMax() {
        data[0] = data[count - 1]
        indices[0] = indices[count - 1]
        count--
        siftDown(0)
    }

    fun sorted(): Pair<PrimitiveArray, IntArray> {
        val sortedData = PrimitiveArray(count)
        val sortedIndices = IntArray(count)

        for (idx in (count - 1) downTo 0) {
            sortedData[idx] = maxValue
            sortedIndices[idx] = maxIndex

            removeMax()
        }

        return sortedData to sortedIndices
    }

    fun clear() {
        count = 0
    }
}

@FilterPrimitives(exclude = [DataType.INT])
internal fun PrimitiveArray.swap(leftIdx: Int, rightIdx: Int) {
    val temp = get(leftIdx)
    this[leftIdx] = this[rightIdx]
    this[rightIdx] = temp
}
