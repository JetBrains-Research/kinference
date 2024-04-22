@file:GeneratePrimitives(
    DataType.NUMBER
)

package io.kinference.utils.inlines

import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import kotlin.jvm.JvmInline
import kotlin.experimental.*

@GenerateNameFromPrimitives
@JvmInline
value class InlinePrimitive(val value: PrimitiveType) {
    inline operator fun plus(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value + other.value).toPrimitive())

    inline operator fun minus(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value - other.value).toPrimitive())

    inline operator fun times(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value * other.value).toPrimitive())

    inline operator fun div(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value / other.value).toPrimitive())

    inline operator fun rem(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value % other.value).toPrimitive())

    @FilterPrimitives(exclude = [DataType.FLOAT, DataType.DOUBLE])
    inline infix fun and(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value and other.value).toPrimitive())
    @FilterPrimitives(exclude = [DataType.FLOAT, DataType.DOUBLE])
    inline infix fun or(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value or other.value).toPrimitive())
    @FilterPrimitives(exclude = [DataType.FLOAT, DataType.DOUBLE])
    inline infix fun xor(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value xor other.value).toPrimitive())
}
