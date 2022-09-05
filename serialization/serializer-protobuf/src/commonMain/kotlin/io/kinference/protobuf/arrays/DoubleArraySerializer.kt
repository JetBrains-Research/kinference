package io.kinference.protobuf.arrays

import io.kinference.protobuf.ProtobufReader

internal class DoubleArrayBuilder(private var data: DoubleArray) : PrimitiveArrayBuilder<DoubleArray>() {
    override var position: Int = data.size
        private set

    init {
        checkCapacity(INITIAL_CAPACITY)
    }

    override fun checkCapacity(requiredCapacity: Int) {
        if (data.size < requiredCapacity)
            data = data.copyOf(requiredCapacity.coerceAtLeast(data.size * 2))
    }

    fun append(element: Double) {
        checkCapacity()
        data[position++] = element
    }

    override fun build() = data.copyOf(position)
}

internal object DoubleArraySerializer : PrimitiveArraySerializer<DoubleArray, DoubleArrayBuilder>() {
    override fun DoubleArray.toBuilder(): DoubleArrayBuilder = DoubleArrayBuilder(this)
    override fun empty(): DoubleArray = DoubleArray(0)

    override fun readElement(reader: ProtobufReader, builder: DoubleArrayBuilder) {
        builder.append(reader.readDouble())
    }
}
