package io.kinference.utils.inlines

import kotlin.jvm.JvmInline

@JvmInline
value class InlineBoolean(val value: Boolean) {
    inline infix fun and(other: InlineBoolean): InlineBoolean = InlineBoolean(this.value && other.value)
    inline infix fun or(other: InlineBoolean): InlineBoolean = InlineBoolean(this.value || other.value)
    inline infix fun xor(other: InlineBoolean): InlineBoolean = InlineBoolean(this.value xor other.value)
}
