package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
import kotlinx.atomicfu.atomic
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

object ArrayDispatcher {
    private val modelDispatchers = mutableMapOf<String, ModelArrayDispatcher>()
    private val mutex = Mutex()

    suspend fun addModelContext(modelContext: String) {
        mutex.withLock {
            modelDispatchers[modelContext] = ModelArrayDispatcher()
        }
    }

    suspend fun removeModelContext(modelContext: String) {
        val modelDispatcher = mutex.withLock {
            modelDispatchers.remove(modelContext)
        }
        modelDispatcher?.close()
    }

    suspend fun addInferenceContext(modelContext: String, inferenceContext: String) {
        modelDispatchers[modelContext]!!.addInferenceContext(inferenceContext)
    }

    suspend fun closeInferenceContext(modelContext: String, inferenceContext: String) {
        modelDispatchers[modelContext]!!.closeInferenceContext(inferenceContext)
    }

    internal suspend fun getArrayContainers(
        type: ArrayTypes,
        size: Int,
        count: Int,
        modelContext: String = NO_MODEL_CONTEXT,
        inferenceContext: String = NO_INFERENCE_CONTEXT
    ): Array<ArrayContainer> {
        if (modelContext == NO_MODEL_CONTEXT || inferenceContext == NO_MODEL_CONTEXT)
            return Array(count) { ArrayContainer(type, size) }

        return modelDispatchers[modelContext]!!.getArrayContainers(inferenceContext, type, size, count)
    }
}

private class ModelArrayDispatcher {
    companion object {
        private const val INIT_SIZE_VALUE: Int = 2
        private val typeSize: Int = ArrayTypes.entries.size
    }

    private val usedArrays: HashMap<String, ConcurrentArrayContainerQueue> = hashMapOf()
    private val unusedArrays: ArrayStorage = ArrayStorage(typeSize, INIT_SIZE_VALUE)
    private val mutex = Mutex()

    class ConcurrentArrayContainerQueue {
        // Initialize the head with the emptyContainer sentinel node
        private var head: ArrayContainer? = ArrayContainer.emptyContainer()
        private var tail: ArrayContainer? = head
        private val isClosed = atomic(false)
        private val lock = atomic(false)

        fun addLast(container: ArrayContainer) {
            while (true) {
                if (lock.compareAndSet(expect = false, update = true)) {
                    if (isClosed.value) {
                        lock.value = false
                        throw IllegalStateException("Cannot add to a closed queue.")
                    }

                    container.next = null
                    tail?.next = container
                    tail = container
                    lock.value = false
                    return
                }
            }
        }

        fun removeFirstOrNull(): ArrayContainer? {
            while (true) {
                if (lock.compareAndSet(expect = false, update = true)) {
                    if (isClosed.value) {
                        lock.value = false
                        throw IllegalStateException("Cannot remove from a closed queue.")
                    }

                    val first = head?.next
                    if (first == null) {
                        lock.value = false
                        return null
                    }

                    head?.next = first.next
                    if (first.next == null) {
                        tail = head
                    }
                    lock.value = false
                    return first
                }
            }
        }

        fun close() {
            while (true) {
                if (lock.compareAndSet(expect = false, update = true)) {
                    isClosed.value = true
                    var current = head
                    while (current != null) {
                        val next = current.next
                        current.next = null
                        current = next
                    }
                    lock.value = false
                    return
                }
            }
        }
    }

    private class ArrayStorage(typeLength: Int, sizeLength: Int) {
        /**
         * Structure is as follows:
         * 1. Array by predefined types (all types are known compiled time)
         * 2. Array by size. Starting with 'INIT_SIZE_VALUE' element and grow it doubling (typically there are no more than 16 different sizes)
         * 3. Queue of array containers (used as FIFO)
         */
        private var storage: Array<Array<ConcurrentArrayContainerQueue>> =
            Array(typeLength) { Array(sizeLength) { ConcurrentArrayContainerQueue() } }

        private var sizeIndices: IntArray = IntArray(typeLength)
        private var sizes: Array<IntArray> = Array(typeLength) { IntArray(sizeLength) }
        private val mutex = Mutex()

        operator fun get(typeIndex: Int, sizeIndex: Int): ConcurrentArrayContainerQueue {
            return storage[typeIndex][sizeIndex]
        }

        suspend fun getArrayContainer(type: ArrayTypes, size: Int): ArrayContainer {
            val tIndex = type.index
            val sIndex = sizes[tIndex].indexOf(size)

            // Checking that we have this array size in our storage for this type
            val idx = if (sIndex != -1) {
                val array = storage[tIndex][sIndex].removeFirstOrNull()
                array?.let {
                    it.marker = ArrayUsageMarker.Used
                    ArrayContainer.resetArray(it)
                    return it
                }
                sIndex
            } else {
                mutex.withLock {
                    if (sizeIndices[tIndex] >= storage[tIndex].size)
                        grow(tIndex)

                    val idx = sizeIndices[tIndex]++
                    sizes[tIndex][idx] = size
                    idx
                }
            }

            return ArrayContainer(type, size, idx)
        }

        fun grow(typeIndex: Int) {
            val newSize = sizes[typeIndex].size * 2
            val newStorage: Array<ConcurrentArrayContainerQueue> = Array(newSize) { ConcurrentArrayContainerQueue() }

            for (i in storage[typeIndex].indices) {
                newStorage[i] = storage[typeIndex][i]
            }

            storage[typeIndex] = newStorage
            sizes[typeIndex] = sizes[typeIndex].copyOf(newSize)
        }

        fun close() {
            for (i in storage.indices) {
                for (j in storage[i].indices) {
                    storage[i][j].close()
                }
            }
        }
    }

    suspend fun addInferenceContext(inferenceContext: String) {
        mutex.withLock {
            usedArrays[inferenceContext] = ConcurrentArrayContainerQueue()
        }
    }

    suspend fun getArrayContainers(inferenceContext: String, type: ArrayTypes, size: Int, count: Int): Array<ArrayContainer> {
        return Array(count) { getArrayContainer(inferenceContext, type, size) }
    }

    suspend fun closeInferenceContext(inferenceContext: String) {
        val usedArrays = mutex.withLock {
            usedArrays.remove(inferenceContext)!!
        }
        var isProcessed = false

        while (!isProcessed) {
            val container = usedArrays.removeFirstOrNull()
            if (container != null) {
                if (container.marker != ArrayUsageMarker.Output) {
                    container.marker = ArrayUsageMarker.Unused
                    unusedArrays[container.arrayTypeIndex, container.arraySizeIndex].addLast(container)
                }
            } else {
                isProcessed = true
            }
        }

        usedArrays.close()
    }

    fun close() {
        unusedArrays.close()
        usedArrays.forEach { it.value.close() }
        usedArrays.clear()
    }

    private suspend fun getArrayContainer(inferenceContext: String, type: ArrayTypes, size: Int): ArrayContainer {
        val newArray = unusedArrays.getArrayContainer(type, size)
        usedArrays[inferenceContext]!!.addLast(newArray)
        return newArray
    }
}
