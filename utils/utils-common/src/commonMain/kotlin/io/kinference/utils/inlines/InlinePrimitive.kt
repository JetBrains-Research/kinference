@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.utils.inlines

import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import kotlin.jvm.JvmInline

@GenerateNameFromPrimitives
@MakePublic
@JvmInline
value class InlinePrimitive(val value: PrimitiveType) {
    operator fun plus(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value + other.value).toPrimitive())
    operator fun minus(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value - other.value).toPrimitive())
    operator fun times(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value * other.value).toPrimitive())
    operator fun div(other: InlinePrimitive): InlinePrimitive = InlinePrimitive((this.value / other.value).toPrimitive())
}
