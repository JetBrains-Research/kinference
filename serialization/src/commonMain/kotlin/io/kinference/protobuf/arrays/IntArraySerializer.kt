package io.kinference.protobuf.arrays

import io.kinference.ndarray.arrays.pointers.IntPointer
import io.kinference.ndarray.arrays.tiled.IntTiledArray
import io.kinference.protobuf.ProtobufReader


internal class IntArrayBuilder(private var data: IntArray) : PrimitiveArrayBuilder<IntArray>() {
    override var position: Int = data.size
        private set

    init {
        checkCapacity(INITIAL_CAPACITY)
    }

    override fun checkCapacity(requiredCapacity: Int) {
        if (data.size < requiredCapacity)
            data = data.copyOf(requiredCapacity.coerceAtLeast(data.size * 2))
    }

    fun append(element: Int) {
        checkCapacity()
        data[position++] = element
    }

    override fun build() = data.copyOf(position)
}

internal object IntArraySerializer : PrimitiveArraySerializer<IntArray, IntArrayBuilder>() {
    override fun IntArray.toBuilder(): IntArrayBuilder = IntArrayBuilder(this)
    override fun empty(): IntArray = IntArray(0)

    override fun readElement(reader: ProtobufReader, builder: IntArrayBuilder) {
        builder.append(reader.readInt())
    }
}

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
