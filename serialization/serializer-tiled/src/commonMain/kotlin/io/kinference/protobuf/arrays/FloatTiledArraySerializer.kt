package io.kinference.protobuf.arrays

import io.kinference.ndarray.arrays.pointers.FloatPointer
import io.kinference.ndarray.arrays.tiled.FloatTiledArray
import io.kinference.protobuf.ProtobufReader

internal class FloatTiledArrayBuilder(data: FloatTiledArray) : ArrayBuilder<FloatTiledArray>() {
    private val pointer = FloatPointer(data)

    fun append(element: Float) {
        pointer.set(element)
        pointer.increment()
    }

    override fun build(): FloatTiledArray {
        require(pointer.linearIndex == pointer.array.size)
        return pointer.array
    }
}

internal object FloatTiledArraySerializer : TiledArraySerializer<FloatTiledArray, FloatTiledArrayBuilder>() {
    override fun empty(shape: IntArray): FloatTiledArray = FloatTiledArray(shape)
    override fun FloatTiledArray.toBuilder(): FloatTiledArrayBuilder = FloatTiledArrayBuilder(this)

    override fun readElement(reader: ProtobufReader, builder: FloatTiledArrayBuilder) {
        builder.append(reader.readFloat())
    }
}
