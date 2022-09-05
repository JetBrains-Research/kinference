package io.kinference.protobuf.arrays

import io.kinference.ndarray.arrays.pointers.IntPointer
import io.kinference.ndarray.arrays.tiled.IntTiledArray
import io.kinference.protobuf.ProtobufReader

internal class IntTiledArrayBuilder(data: IntTiledArray) : ArrayBuilder<IntTiledArray>() {
    private val pointer = IntPointer(data)

    fun append(element: Int) {
        pointer.set(element)
        pointer.increment()
    }

    override fun build(): IntTiledArray {
        require(pointer.linearIndex == pointer.array.size)
        return pointer.array
    }
}

internal object IntTiledArraySerializer : TiledArraySerializer<IntTiledArray, IntTiledArrayBuilder>() {
    override fun empty(shape: IntArray): IntTiledArray = IntTiledArray(shape)
    override fun IntTiledArray.toBuilder(): IntTiledArrayBuilder = IntTiledArrayBuilder(this)

    override fun readElement(reader: ProtobufReader, builder: IntTiledArrayBuilder) {
        builder.append(reader.readInt())
    }
}
