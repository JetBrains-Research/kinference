package io.kinference.ndarray.inlines

import kotlin.jvm.JvmInline

@JvmInline
value class InlineBoolean(val value: Boolean) {
    infix fun and(other: InlineBoolean): InlineBoolean = InlineBoolean(this.value && other.value)
    infix fun or(other: InlineBoolean): InlineBoolean = InlineBoolean(this.value || other.value)
    infix fun xor(other: InlineBoolean): InlineBoolean = InlineBoolean(this.value xor other.value)
}
