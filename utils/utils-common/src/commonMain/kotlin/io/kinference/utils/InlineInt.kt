package io.kinference.utils

import kotlin.jvm.JvmInline

@JvmInline
value class InlineInt(val value: Int) {
    inline operator fun plus(other: InlineInt): InlineInt = InlineInt((this.value + other.value).toInt())

    inline operator fun minus(other: InlineInt): InlineInt = InlineInt((this.value - other.value).toInt())

    inline operator fun times(other: InlineInt): InlineInt = InlineInt((this.value * other.value).toInt())

    inline operator fun div(other: InlineInt): InlineInt = InlineInt((this.value / other.value).toInt())

    inline operator fun rem(other: InlineInt): InlineInt = InlineInt((this.value % other.value).toInt())

    inline infix fun and(other: InlineInt): InlineInt = InlineInt((this.value and other.value).toInt())

    inline infix fun or(other: InlineInt): InlineInt = InlineInt((this.value or other.value).toInt())

    inline infix fun xor(other: InlineInt): InlineInt = InlineInt((this.value xor other.value).toInt())
}

