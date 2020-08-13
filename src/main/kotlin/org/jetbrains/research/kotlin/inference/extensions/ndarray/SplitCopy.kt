package org.jetbrains.research.kotlin.inference.extensions.ndarray

fun FloatArray.copySplitFragment(srcOffset: Int, numFragments: Int, sliceLen: Int, fragOffset: Int): FloatArray {
    if (fragOffset == sliceLen) {
        return this.copyOfRange(0, numFragments * sliceLen)
    }

    val dst = FloatArray(sliceLen * numFragments)
    repeat(numFragments) {
        val start = srcOffset + fragOffset * it
        this.copyInto(dst, it * sliceLen, start, start + sliceLen)
    }
    return dst
}

fun LongArray.copySplitFragment(srcOffset: Int, numFragments: Int, sliceLen: Int, fragOffset: Int): LongArray {
    if (fragOffset == sliceLen) {
        return this.copyOfRange(0, numFragments * sliceLen)
    }

    val dst = LongArray(sliceLen * numFragments)
    repeat(numFragments) {
        val start = srcOffset + fragOffset * it
        this.copyInto(dst, it * sliceLen, start, start + sliceLen)
    }
    return dst
}

fun DoubleArray.copySplitFragment(srcOffset: Int, numFragments: Int, sliceLen: Int, fragOffset: Int): DoubleArray {
    if (fragOffset == sliceLen) {
        return this.copyOfRange(0, numFragments * sliceLen)
    }

    val dst = DoubleArray(sliceLen * numFragments)
    repeat(numFragments) {
        val start = srcOffset + fragOffset * it
        this.copyInto(dst, it * sliceLen, start, start + sliceLen)
    }
    return dst
}

fun ShortArray.copySplitFragment(srcOffset: Int, numFragments: Int, sliceLen: Int, fragOffset: Int): ShortArray {
    if (fragOffset == sliceLen) {
        return this.copyOfRange(0, numFragments * sliceLen)
    }

    val dst = ShortArray(sliceLen * numFragments)
    repeat(numFragments) {
        val start = srcOffset + fragOffset * it
        this.copyInto(dst, it * sliceLen, start, start + sliceLen)
    }
    return dst
}

fun IntArray.copySplitFragment(srcOffset: Int, numFragments: Int, sliceLen: Int, fragOffset: Int): IntArray {
    if (fragOffset == sliceLen) {
        return this.copyOfRange(0, numFragments * sliceLen)
    }

    val dst = IntArray(sliceLen * numFragments)
    repeat(numFragments) {
        val start = srcOffset + fragOffset * it
        this.copyInto(dst, it * sliceLen, start, start + sliceLen)
    }
    return dst
}
