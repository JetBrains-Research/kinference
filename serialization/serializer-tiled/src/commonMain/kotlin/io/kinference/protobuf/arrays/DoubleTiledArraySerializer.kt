package io.kinference.protobuf.arrays

import io.kinference.ndarray.arrays.pointers.DoublePointer
import io.kinference.ndarray.arrays.tiled.DoubleTiledArray
import io.kinference.protobuf.ProtobufReader

internal class DoubleTiledArrayBuilder(data: DoubleTiledArray) : ArrayBuilder<DoubleTiledArray>() {
    private val pointer = DoublePointer(data)

    fun append(element: Double) {
        pointer.set(element)
        pointer.increment()
    }

    override fun build(): DoubleTiledArray {
        require(pointer.linearIndex == pointer.array.size)
        return pointer.array
    }
}

internal object DoubleTiledArraySerializer : TiledArraySerializer<DoubleTiledArray, DoubleTiledArrayBuilder>() {
    override fun empty(shape: IntArray): DoubleTiledArray = DoubleTiledArray(shape)
    override fun DoubleTiledArray.toBuilder(): DoubleTiledArrayBuilder = DoubleTiledArrayBuilder(this)

    override fun readElement(reader: ProtobufReader, builder: DoubleTiledArrayBuilder) {
        builder.append(reader.readDouble())
    }
}
