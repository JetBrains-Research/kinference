package io.kinference.utils

typealias StateMarker = (ArrayUsageMarker) -> Unit

enum class ArrayUsageMarker {
    Unused,
    Used,
    ContextOutput,
    GlobalOutput
}

class ArrayContainer<T>(val array: T, var marker: ArrayUsageMarker = ArrayUsageMarker.Used) {
    val markAsOutput: StateMarker = {
        marker = it
    }
}

object ArraysDispatcher {
    private const val INIT_SIZE_VALUE: Int = 1
    private val typesSize = ArrayTypes.entries.size
    private val contextStack: ArrayDeque<String> = ArrayDeque()
    private var currentOperatorContext: String = "NotAnOperator"
    private val contexts: MutableSet<String> = mutableSetOf(currentOperatorContext)

    private var contextUsedArrays: Array<Array<MutableMap<String, ArrayDeque<ArrayContainer<*>>>>> =
        Array(typesSize) { Array(INIT_SIZE_VALUE) { mutableMapOf() } }
    private var contextUnusedArrays: Array<Array<MutableMap<String, ArrayDeque<ArrayContainer<*>>>>> =
        Array(typesSize) { Array(INIT_SIZE_VALUE) { mutableMapOf() } }

    private val sizeIndices = IntArray(typesSize)
    private var sizes = Array(typesSize) { IntArray(INIT_SIZE_VALUE) }
    private var sizesUsage = Array(typesSize) { IntArray(INIT_SIZE_VALUE) }

    fun addContexts(operators: List<String>) {
        if (currentOperatorContext == "NotAnOperator") {
            contexts.addAll(operators)
        } else {
            operators.forEach { contexts.add("$currentOperatorContext.$it") }
        }

        for (i in 0 until typesSize) {
            for (j in 0 until sizeIndices[i]) {
                contexts.forEach {
                    if (contextUsedArrays[i][j][it] == null)
                        contextUsedArrays[i][j][it] = ArrayDeque()
                    if (contextUnusedArrays[i][j][it] == null)
                        contextUnusedArrays[i][j][it] = ArrayDeque()
                }
            }
        }
    }

    fun setOperatorContext(context: String) {
        pushOperatorContext(context)
    }

    inline fun <reified T> getArraysAndMarkers(type: ArrayTypes, size: Int, count: Int): Array<ArrayContainer<T>> {
        return Array(count) { getArray(type, size) as ArrayContainer<T> }
    }

    fun getArray(type: ArrayTypes, size: Int): ArrayContainer<*> {
        val tIndex = type.index
        val used = contextUnusedArrays[tIndex]
        val sIndex = sizes[tIndex].indexOf(size)
        return if (sIndex != -1 && used[sIndex][currentOperatorContext]!!.isNotEmpty()) {
            val array = used[sIndex][currentOperatorContext]!!.removeFirst()
            array.marker = ArrayUsageMarker.Used
            resetPrimitiveArray(array)
            contextUsedArrays[tIndex][sIndex][currentOperatorContext]!!.addLast(array)
            sizesUsage[tIndex][sIndex]++
            array
        } else {
            val newArray = type.createArray(size)
            putArray(type, size, newArray)
            newArray
        }
    }

    fun releaseUsedInContext() {
        // When the context is released, move used arrays to unused struct and skip context outputs
        for (i in 0 until typesSize) {
            for (j in 0 until sizeIndices[i]) {
                val usedArrays = contextUsedArrays[i][j][currentOperatorContext]!!
                val unusedArrays = contextUnusedArrays[i][j][currentOperatorContext]!!

                for (k in usedArrays.size - 1 downTo 0) {
                    val arrayContainer = usedArrays[k]
                    if (arrayContainer.marker != ArrayUsageMarker.ContextOutput) {
                        unusedArrays.addLast(arrayContainer)
                        usedArrays.removeAt(k)
                    }
                }
            }
        }

        popOperatorContext()
    }

    fun releaseAllOutputArrays() {
        // Iterate through all types, sizes, and contexts
        for (i in 0 until typesSize) {
            for (j in 0 until sizeIndices[i]) {
                val usedPerSize = contextUsedArrays[i][j]
                val unusedPerSize = contextUnusedArrays[i][j]

                contexts.forEach { context ->
                    val usedArrays = usedPerSize[context]!!
                    val unusedArrays = unusedPerSize[context]!!

                    // Move all context outputs into unused and permanently remove all global outputs
                    for (k in usedArrays.size - 1 downTo 0) {
                        val arrayContainer = usedArrays[k]
                        if (arrayContainer.marker == ArrayUsageMarker.ContextOutput) {
                            arrayContainer.marker = ArrayUsageMarker.Unused
                            unusedArrays.addLast(arrayContainer)
                            usedArrays.removeAt(k)
                        } else if (arrayContainer.marker == ArrayUsageMarker.GlobalOutput) {
                            arrayContainer.marker = ArrayUsageMarker.Unused
                            usedArrays.removeAt(k)
                        }
                    }
                }
            }
        }
        reorderBasedOnUsage()
    }

    private fun pushOperatorContext(newContext: String) {
        contextStack.addFirst(currentOperatorContext)  // Save the current context
        currentOperatorContext = if (currentOperatorContext == "NotAnOperator") {
            newContext  // If the base context, start new
        } else {
            "$currentOperatorContext.$newContext"  // Otherwise, append
        }
    }

    private fun popOperatorContext() {
        currentOperatorContext = if (contextStack.isNotEmpty()) {
            contextStack.removeFirst()
        } else {
            "NotAnOperator"
        }
    }

    private fun putArray(type: ArrayTypes, size: Int, array: ArrayContainer<*>) {
        val tIndex = type.index
        val used = contextUnusedArrays[tIndex]
        var idx = sizes[tIndex].indexOf(size)
        if (idx == -1) {
            if (sizeIndices[tIndex] >= used.size)
                grow(type)

            idx = sizeIndices[tIndex]++
            for (i in idx until sizes[tIndex].size) {
                contexts.forEach { context ->
                    contextUsedArrays[tIndex][i][context] = ArrayDeque()
                    contextUnusedArrays[tIndex][i][context] = ArrayDeque()
                }
            }
            sizes[tIndex][idx] = size
        }
        contextUsedArrays[tIndex][idx][currentOperatorContext]!!.addLast(array)
    }

    private fun grow(type: ArrayTypes) {
        // Determine the new size, typically double the current size
        val tIndex = type.index
        val newSize = sizes[tIndex].size * 2

        // Create new arrays of the new size
        val newContextUsedArrays = Array(typesSize) { Array(newSize) { mutableMapOf<String, ArrayDeque<ArrayContainer<*>>>() } }
        val newContextUnusedArrays = Array(typesSize) { Array(newSize) { mutableMapOf<String, ArrayDeque<ArrayContainer<*>>>() } }

        // Transfer the old data into the new arrays
        for (i in contextUsedArrays[tIndex].indices) {
            newContextUsedArrays[tIndex][i] = contextUsedArrays[tIndex][i]
            newContextUnusedArrays[tIndex][i] = contextUnusedArrays[tIndex][i]
        }

        // Assign the new arrays back to the contextUsedArrays and contextUnusedArrays
        contextUsedArrays[tIndex] = newContextUsedArrays[tIndex]
        contextUnusedArrays[tIndex] = newContextUnusedArrays[tIndex]

        // Resize the sizes array
        sizes[tIndex] = sizes[tIndex].copyOf(newSize)
        sizesUsage[tIndex] = sizesUsage[tIndex].copyOf(newSize)
    }

    private fun reorderBasedOnUsage() {
        for (typeIndex in 0 until typesSize) {
            val currentOrder = sizes[typeIndex].indices.toList()

            val sortedIndices = sizes[typeIndex].indices
                .map { index -> Pair(index, sizesUsage[typeIndex].getOrElse(index) { 0 }) }
                .sortedWith(compareByDescending<Pair<Int, Int>> { it.second }.thenBy { sizes[typeIndex][it.first] == 0 })

            // Check if reordering is necessary
            val newOrder = sortedIndices.map { it.first }
            if (newOrder == currentOrder) continue

            // Apply changes only if there's a different order
            val newSizes = IntArray(sizes[typeIndex].size)
            val newSizesUsage = IntArray(sizes[typeIndex].size)
            val newContextUsed = Array(sizes[typeIndex].size) { mutableMapOf<String, ArrayDeque<ArrayContainer<*>>>() }
            val newContextUnused = Array(sizes[typeIndex].size) { mutableMapOf<String, ArrayDeque<ArrayContainer<*>>>() }

            for (i in sortedIndices.indices) {
                val oldIndex = sortedIndices[i].first
                newSizes[i] = sizes[typeIndex][oldIndex]
                newSizesUsage[i] = sizesUsage[typeIndex][oldIndex]
                newContextUsed[i] = contextUsedArrays[typeIndex][oldIndex]
                newContextUnused[i] = contextUnusedArrays[typeIndex][oldIndex]
            }

            sizes[typeIndex] = newSizes
            sizesUsage[typeIndex] = newSizesUsage
            contextUsedArrays[typeIndex] = newContextUsed
            contextUnusedArrays[typeIndex] = newContextUnused
        }

        // Reset or decay sizesUsage here if necessary
        decaySizesUsage()
    }

    private fun decaySizesUsage() {
        // Decay mechanism here is simple halving of usage counts
        for (typeIndex in 0 until typesSize) {
            for (i in sizesUsage[typeIndex].indices) {
                sizesUsage[typeIndex][i] /= 2
            }
        }
    }
}

enum class ArrayTypes(val index: Int, val initializer: (Int) -> ArrayContainer<*>) {
    ByteArray(0, { size -> ArrayContainer(ByteArray(size)) }),
    UByteArray(1, { size -> ArrayContainer(UByteArray(size)) }),
    ShortArray(2, { size -> ArrayContainer(ShortArray(size)) }),
    UShortArray(3, { size -> ArrayContainer(UShortArray(size)) }),
    IntArray(4, { size -> ArrayContainer(IntArray(size)) }),
    UIntArray(5, { size -> ArrayContainer(UIntArray(size)) }),
    LongArray(6, { size -> ArrayContainer(LongArray(size)) }),
    ULongArray(7, { size -> ArrayContainer(ULongArray(size)) }),
    FloatArray(8, { size -> ArrayContainer(FloatArray(size)) }),
    DoubleArray(9, { size -> ArrayContainer(DoubleArray(size)) }),
    CharArray(10, { size -> ArrayContainer(CharArray(size)) }),
    BooleanArray(11, { size -> ArrayContainer(BooleanArray(size)) });

    fun createArray(size: Int) : ArrayContainer<*> {
        return initializer(size)
    }
}

fun resetPrimitiveArray(array: ArrayContainer<*>) {
    when (val arr = array.array!!) {
        is ByteArray -> arr.fill(0)       // 8-bit signed
        is UByteArray -> arr.fill(0u)     // 8-bit unsigned
        is ShortArray -> arr.fill(0)      // 16-bit signed
        is UShortArray -> arr.fill(0u)    // 16-bit unsigned
        is IntArray -> arr.fill(0)        // 32-bit signed
        is UIntArray -> arr.fill(0u)      // 32-bit unsigned
        is LongArray -> arr.fill(0L)      // 64-bit signed
        is ULongArray -> arr.fill(0U)     // 64-bit unsigned
        is FloatArray -> arr.fill(0.0f)
        is DoubleArray -> arr.fill(0.0)
        is CharArray -> arr.fill('\u0000')
        is BooleanArray -> arr.fill(false)
        else -> throw IllegalArgumentException("Unsupported array type")
    }
}
