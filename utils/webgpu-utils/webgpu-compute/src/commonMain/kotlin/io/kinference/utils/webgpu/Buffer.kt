package io.kinference.utils.webgpu

expect class Buffer {
    val size: Int

    fun destroy()
    fun getMappedRange(offset: Int = 0, size: Int = maxOf(0, this.size - offset)): BufferData
    // FIXME size: Int = maxOf(0, this.size - offset) is incorrectly translated to JS
    suspend fun mapAsync(mode: MapModeFlags, offset: Int = 0, size: Int = -1)
    fun unmap()
}
