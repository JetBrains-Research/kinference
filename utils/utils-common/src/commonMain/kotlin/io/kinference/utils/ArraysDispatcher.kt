package io.kinference.utils

class ArrayWrapper(val array: Any, var isInUse: Boolean = true)

object ArraysDispatcher {
    private var currentOperatorContext: String = "NotAnOperator"
    private val contextArrayMap: MutableMap<String, MutableMap<String, MutableMap<Int, MutableList<ArrayWrapper>>>> = mutableMapOf()
    private val contextUnusedArrays: MutableMap<String, MutableMap<String, MutableMap<Int, ArrayDeque<ArrayWrapper>>>> = mutableMapOf()
    private val contextOutputArrays: MutableMap<String, MutableSet<Any>> = mutableMapOf()

    fun setOperatorContext(context: String) {
        currentOperatorContext = context
    }

    fun getArray(type: String, size: Int): Any? {
        val unusedList = contextUnusedArrays[currentOperatorContext]?.get(type)?.get(size)
        val wrapper = unusedList?.removeFirstOrNull()
        if (wrapper != null) {
            wrapper.isInUse = true
            resetPrimitiveArray(wrapper.array)
            return wrapper.array
        }
        // No unused array available, create a new one if necessary
        return null
    }

    fun putArray(type: String, size: Int, array: Any) {
        val wrapper = ArrayWrapper(array)
        contextArrayMap.getOrPut(currentOperatorContext) { mutableMapOf() }
            .getOrPut(type) { mutableMapOf() }
            .getOrPut(size) { mutableListOf() }
            .add(wrapper)
        // Do not add to unused as it's in use right away
    }

    fun releaseOutputArrays() {
        contextOutputArrays[currentOperatorContext]?.clear()
    }

    fun releaseAllOutputArrays() {
        contextOutputArrays.clear()
    }

    //TODO: we need to track cycles of output tensors usage
    fun markOutput(array: Any) {
        val arrays = contextOutputArrays.getOrPut(currentOperatorContext) { -> mutableSetOf() }
        arrays.add(array)
    }

    fun releaseContext() {
        // When the context is released, mark all arrays as unused
        contextArrayMap[currentOperatorContext]?.forEach { (type, typeMap) ->
            typeMap.forEach { (size, wrappers) ->
                wrappers.forEach { wrapper ->
                    if (wrapper.isInUse && !contextOutputArrays[currentOperatorContext]?.contains(wrapper.array)!!) {
                        wrapper.isInUse = false
                        contextUnusedArrays.getOrPut(currentOperatorContext) { mutableMapOf() }
                            .getOrPut(type) { mutableMapOf() }
                            .getOrPut(size) { ArrayDeque() }
                            .addLast(wrapper)
                    }
                }
            }
        }

        currentOperatorContext = "NotAnOperator"
    }
}

fun resetPrimitiveArray(array: Any) {
    when (array) {
        is ByteArray -> array.fill(0)       // 8 bit signed
        is UByteArray -> array.fill(0u)     // 8 bit unsigned
        is ShortArray -> array.fill(0)      // 16 bit signed
        is UShortArray -> array.fill(0u)    // 16 bit unsigned
        is IntArray -> array.fill(0)        // 32 bit signed
        is UIntArray -> array.fill(0u)      // 32 bit unsigned
        is LongArray -> array.fill(0L)      // 64 bit signed
        is ULongArray -> array.fill(0U)     // 64 bit unsigned
        is FloatArray -> array.fill(0.0f)
        is DoubleArray -> array.fill(0.0)
        is CharArray -> array.fill('\u0000')
        is BooleanArray -> array.fill(false)
        else -> throw IllegalArgumentException("Unsupported array type")
    }
}
