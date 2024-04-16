@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.LONG,
    DataType.UBYTE,
    DataType.USHORT,
    DataType.UINT,
    DataType.ULONG,
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.inlines

import io.kinference.ndarray.stubs.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import kotlin.experimental.*
import kotlin.jvm.JvmInline

@GenerateNameFromPrimitives
@MakePublic
@JvmInline
internal value class InlinePrimitive(val value: PrimitiveType) {
    operator fun plus(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value + other.value).toPrimitive())

    operator fun minus(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value - other.value).toPrimitive())

    operator fun times(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value * other.value).toPrimitive())

    operator fun div(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value / other.value).toPrimitive())

    operator fun rem(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value % other.value).toPrimitive())

    @FilterPrimitives(exclude = [DataType.FLOAT, DataType.DOUBLE])
    infix fun and(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value and other.value).toPrimitive())
    @FilterPrimitives(exclude = [DataType.FLOAT, DataType.DOUBLE])
    infix fun or(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value or other.value).toPrimitive())
    @FilterPrimitives(exclude = [DataType.FLOAT, DataType.DOUBLE])
    infix fun xor(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value xor other.value).toPrimitive())
}
