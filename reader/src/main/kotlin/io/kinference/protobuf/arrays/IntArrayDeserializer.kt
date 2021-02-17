package io.kinference.protobuf.arrays

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

internal object IntArrayDeserializer : PrimitiveArraySerializer<IntArray, IntArrayBuilder>() {
    override fun IntArray.toBuilder(): IntArrayBuilder = IntArrayBuilder(this)
    override fun empty(): IntArray = IntArray(0)

    override fun readElement(reader: ProtobufReader, builder: IntArrayBuilder) {
        builder.append(reader.readInt())
    }
}
