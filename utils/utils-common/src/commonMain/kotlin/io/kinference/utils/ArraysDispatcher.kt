package io.kinference.utils


enum class ArrayUsageMarker {
    Unused,
    Used,
    ContextOutput,
    GlobalOutput
}

class ArrayContainer<T>(val array: T, var marker: ArrayUsageMarker = ArrayUsageMarker.Used) {
    val markAsOutput: (ArrayUsageMarker) -> Unit = {
        marker = it
    }
}

object ArraysDispatcher {
    private const val INIT_SIZE_VALUE: Int = 16
    private val contextStack: ArrayDeque<String> = ArrayDeque()
    private var currentOperatorContext: String = "NotAnOperator"
    private val contexts: MutableSet<String> = mutableSetOf(currentOperatorContext)

    private var contextUsedArrays: Array<Array<MutableMap<String, ArrayDeque<ArrayContainer<*>>>>> =
        Array(INIT_SIZE_VALUE) { Array(ArrayTypes.entries.size) { mutableMapOf() } }
    private var contextUnusedArrays: Array<Array<MutableMap<String, ArrayDeque<ArrayContainer<*>>>>> =
        Array(INIT_SIZE_VALUE) { Array(ArrayTypes.entries.size) { mutableMapOf() } }

    private var sIdx = 0
    private var sizes = IntArray(INIT_SIZE_VALUE)

    fun addContexts(operators: List<String>) {
        if (currentOperatorContext == "NotAnOperator") {
            contexts.addAll(operators)
        } else {
            operators.forEach { contexts.add("$currentOperatorContext.$it") }
        }

        for (i in 0 until sIdx) {
            contexts.forEach { context ->
                contextUsedArrays[i].forEach {
                    if (it[context] == null)
                        it[context] = ArrayDeque()
                }
                contextUnusedArrays[i].forEach {
                    if (it[context] == null)
                        it[context] = ArrayDeque()
                }
            }
        }
    }

    fun setOperatorContext(context: String) {
        pushOperatorContext(context)
//        currentOperatorContext = context
    }

    inline fun <reified T> getArraysAndMarkers(type: ArrayTypes, size: Int, count: Int): Pair<Array<T>, Array<(ArrayUsageMarker) -> Unit>> {
        val arrays = Array(count) { type.zeroes as T }
        val markers = Array(count) { { _: ArrayUsageMarker -> } }

        // Populate the arrays and markers
        for (i in 0 until count) {
            val container = getArray(type, size)
            container.marker = ArrayUsageMarker.Used
            arrays[i] = container.array as T
            markers[i] = container.markAsOutput
        }

        return Pair(arrays, markers)
    }


    fun getArray(type: ArrayTypes, size: Int): ArrayContainer<*> {
        val idx = sizes.indexOf(size)
        if (idx == -1) {
            val newArray = type.createArray(size)
            putArray(type, size, newArray)
            return newArray
        } else {
            val contextUnused = contextUnusedArrays[idx][type.index][currentOperatorContext]!!
            if (contextUnused.isNotEmpty()) {
                val array = contextUnused.removeFirst()
                resetPrimitiveArray(array)
                contextUsedArrays[idx][type.index][currentOperatorContext]!!.addLast(array)
                return array
            } else {
                val newArray = type.createArray(size)
                putArray(type, size, newArray)
                return newArray
            }
        }
    }

    private fun putArray(type: ArrayTypes, size: Int, array: ArrayContainer<*>) {
        var idx = sizes.indexOf(size)
        if (idx == -1) {
            if (sIdx >= contextUsedArrays.size)
                grow()

            idx = sIdx++
            contexts.forEach { context ->
                contextUsedArrays[idx].forEach { it[context] = ArrayDeque() }
                contextUnusedArrays[idx].forEach { it[context] = ArrayDeque() }
            }
            sizes[idx] = size
        }
        contextUsedArrays[idx][type.index][currentOperatorContext]!!.addLast(array)
    }

    private fun grow() {
        // Determine the new size, typically double the current size
        val newSize = sizes.size * 2

        // Create new arrays of the new size
        val newContextUsedArrays = Array(newSize) { Array(ArrayTypes.entries.size) { mutableMapOf<String, ArrayDeque<ArrayContainer<*>>>() } }
        val newContextUnusedArrays = Array(newSize) { Array(ArrayTypes.entries.size) { mutableMapOf<String, ArrayDeque<ArrayContainer<*>>>() } }

        // Transfer the old data into the new arrays
        for (i in contextUsedArrays.indices) {
            newContextUsedArrays[i] = contextUsedArrays[i]
            newContextUnusedArrays[i] = contextUnusedArrays[i]
        }

        // Assign the new arrays back to the contextUsedArrays and contextUnusedArrays
        contextUsedArrays = newContextUsedArrays
        contextUnusedArrays = newContextUnusedArrays

        // Resize the sizes array
        sizes = sizes.copyOf(newSize)
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

    fun releaseAllOutputArrays() {
        // Iterate through all sizes, types, and contexts
        for (i in 0 until sIdx) {
            val usedPerSize = contextUsedArrays[i]
            val unusedPerSize = contextUnusedArrays[i]

            for (j in usedPerSize.indices) {
                contexts.forEach { context ->
                    val usedArrays = usedPerSize[j][context]!!
                    val unusedArrays = unusedPerSize[j][context]!!

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
    }

    fun releaseContext() {
        // When the context is released, move all arrays except output to unused struct
        for (i in 0 until sIdx) {
            val usedPerSize = contextUsedArrays[i]
            val unusedPerSize = contextUnusedArrays[i]

            for (j in usedPerSize.indices) {
                val usedArrays = usedPerSize[j][currentOperatorContext]!!
                val unusedArrays = unusedPerSize[j][currentOperatorContext]!!

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

//        currentOperatorContext = "NotAnOperator"
    }
}

enum class ArrayTypes(val index: Int, val initializer: (Int) -> ArrayContainer<*>, val zeroes: Any) {
    ByteArray(0, { size -> ArrayContainer(ByteArray(size)) }, ByteArray(0)),
    UByteArray(1, { size -> ArrayContainer(UByteArray(size)) }, UByteArray(0)),
    ShortArray(2, { size -> ArrayContainer(ShortArray(size)) }, ShortArray(0)),
    UShortArray(3, { size -> ArrayContainer(UShortArray(size)) }, UShortArray(0)),
    IntArray(4, { size -> ArrayContainer(IntArray(size)) }, IntArray(0)),
    UIntArray(5, { size -> ArrayContainer(UIntArray(size)) }, UIntArray(0)),
    LongArray(6, { size -> ArrayContainer(LongArray(size)) }, LongArray(0)),
    ULongArray(7, { size -> ArrayContainer(ULongArray(size)) }, ULongArray(0)),
    FloatArray(8, { size -> ArrayContainer(FloatArray(size)) }, FloatArray(0)),
    DoubleArray(9, { size -> ArrayContainer(DoubleArray(size)) }, DoubleArray(0)),
    CharArray(10, { size -> ArrayContainer(CharArray(size)) }, CharArray(0)),
    BooleanArray(11, { size -> ArrayContainer(BooleanArray(size)) }, BooleanArray(0));

    fun createArray(size: Int) : ArrayContainer<*> {
        return initializer(size)
    }
}

fun resetPrimitiveArray(array: ArrayContainer<*>) {
    when (val arr = array.array!!) {
        is ByteArray -> arr.fill(0)       // 8 bit signed
        is UByteArray -> arr.fill(0u)     // 8 bit unsigned
        is ShortArray -> arr.fill(0)      // 16 bit signed
        is UShortArray -> arr.fill(0u)    // 16 bit unsigned
        is IntArray -> arr.fill(0)        // 32 bit signed
        is UIntArray -> arr.fill(0u)      // 32 bit unsigned
        is LongArray -> arr.fill(0L)      // 64 bit signed
        is ULongArray -> arr.fill(0U)     // 64 bit unsigned
        is FloatArray -> arr.fill(0.0f)
        is DoubleArray -> arr.fill(0.0)
        is CharArray -> arr.fill('\u0000')
        is BooleanArray -> arr.fill(false)
        else -> throw IllegalArgumentException("Unsupported array type")
    }
}
