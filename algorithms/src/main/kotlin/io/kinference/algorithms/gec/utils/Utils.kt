package io.kinference.algorithms.gec.utils

fun IntRange.withOffset(offset: Int) = IntRange(this.start + offset, this.endInclusive + offset)
