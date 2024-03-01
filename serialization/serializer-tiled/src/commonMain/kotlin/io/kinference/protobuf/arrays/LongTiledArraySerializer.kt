package io.kinference.protobuf.arrays

import io.kinference.ndarray.arrays.pointers.LongPointer
import io.kinference.ndarray.arrays.tiled.LongTiledArray
import io.kinference.protobuf.ProtobufReader

internal class LongTiledArrayBuilder(data: LongTiledArray) : ArrayBuilder<LongTiledArray>() {
    private val pointer = LongPointer(data)

    fun append(element: Long) {
        pointer.set(element)
        pointer.increment()
    }

    override fun build(): LongTiledArray {
        require(pointer.linearIndex == pointer.array.size)
        return pointer.array
    }
}

internal object LongTiledArraySerializer : TiledArraySerializer<LongTiledArray, LongTiledArrayBuilder>() {
    override suspend fun empty(shape: IntArray): LongTiledArray = LongTiledArray(shape)
    override fun LongTiledArray.toBuilder(): LongTiledArrayBuilder = LongTiledArrayBuilder(this)

    override fun readElement(reader: ProtobufReader, builder: LongTiledArrayBuilder) {
        builder.append(reader.readLong())
    }
}

internal object ULongTiledArraySerializer : TiledArraySerializer<LongTiledArray, LongTiledArrayBuilder>() {
    override suspend fun empty(shape: IntArray): LongTiledArray = LongTiledArray(shape)
    override fun LongTiledArray.toBuilder(): LongTiledArrayBuilder = LongTiledArrayBuilder(this)

    override fun readElement(reader: ProtobufReader, builder: LongTiledArrayBuilder) {
        builder.append(reader.readULong())
    }
}
