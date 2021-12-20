package io.kinference.utils.webgpu

expect class BufferData {
    val size: Int

    constructor(array: IntArray)
    constructor(array: FloatArray)

    fun set(data: BufferData)
    fun set(array: IntArray)
    fun set(array: FloatArray)

    fun toIntArray(): IntArray
    fun toUIntArray(): UIntArray
    fun toFloatArray(): FloatArray
}
