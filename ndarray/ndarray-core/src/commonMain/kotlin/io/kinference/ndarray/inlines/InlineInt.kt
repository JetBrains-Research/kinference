package io.kinference.ndarray.inlines

import kotlin.jvm.JvmInline

@JvmInline
value class InlineInt(val value: Int) {
    operator fun plus(other: InlineInt): InlineInt = InlineInt((this.value + other.value).toInt())

    operator fun minus(other: InlineInt): InlineInt = InlineInt((this.value - other.value).toInt())

    operator fun times(other: InlineInt): InlineInt = InlineInt((this.value * other.value).toInt())

    operator fun div(other: InlineInt): InlineInt = InlineInt((this.value / other.value).toInt())

    operator fun rem(other: InlineInt): InlineInt = InlineInt((this.value % other.value).toInt())

    infix fun and(other: InlineInt): InlineInt = InlineInt((this.value and other.value).toInt())

    infix fun or(other: InlineInt): InlineInt = InlineInt((this.value or other.value).toInt())

    infix fun xor(other: InlineInt): InlineInt = InlineInt((this.value xor other.value).toInt())
}

