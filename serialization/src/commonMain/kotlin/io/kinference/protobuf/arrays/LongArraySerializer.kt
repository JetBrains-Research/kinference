package io.kinference.protobuf.arrays

import io.kinference.ndarray.arrays.pointers.LongPointer
import io.kinference.ndarray.arrays.tiled.LongTiledArray
import io.kinference.protobuf.ProtobufReader


internal class LongArrayBuilder(private var data: LongArray) : PrimitiveArrayBuilder<LongArray>() {
    override var position: Int = data.size
        private set

    init {
        checkCapacity(INITIAL_CAPACITY)
    }

    override fun checkCapacity(requiredCapacity: Int) {
        if (data.size < requiredCapacity)
            data = data.copyOf(requiredCapacity.coerceAtLeast(data.size * 2))
    }

    fun append(element: Long) {
        checkCapacity()
        data[position++] = element
    }

    override fun build() = data.copyOf(position)
}

internal object LongArraySerializer : PrimitiveArraySerializer<LongArray, LongArrayBuilder>() {
    override fun LongArray.toBuilder(): LongArrayBuilder = LongArrayBuilder(this)
    override fun empty(): LongArray = LongArray(0)

    override fun readElement(reader: ProtobufReader, builder: LongArrayBuilder) {
        builder.append(reader.readLong())
    }
}

internal object ULongArraySerializer : PrimitiveArraySerializer<LongArray, LongArrayBuilder>() {
    override fun LongArray.toBuilder(): LongArrayBuilder = LongArrayBuilder(this)
    override fun empty(): LongArray = LongArray(0)

    override fun readElement(reader: ProtobufReader, builder: LongArrayBuilder) {
        builder.append(reader.readULong())
    }
}

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
    override fun empty(shape: IntArray): LongTiledArray = LongTiledArray(shape)
    override fun LongTiledArray.toBuilder(): LongTiledArrayBuilder = LongTiledArrayBuilder(this)

    override fun readElement(reader: ProtobufReader, builder: LongTiledArrayBuilder) {
        builder.append(reader.readLong())
    }
}

internal object ULongTiledArraySerializer : TiledArraySerializer<LongTiledArray, LongTiledArrayBuilder>() {
    override fun empty(shape: IntArray): LongTiledArray = LongTiledArray(shape)
    override fun LongTiledArray.toBuilder(): LongTiledArrayBuilder = LongTiledArrayBuilder(this)

    override fun readElement(reader: ProtobufReader, builder: LongTiledArrayBuilder) {
        builder.append(reader.readULong())
    }
}
