package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.dataInt
import kotlin.test.Test
import kotlinx.coroutines.test.runTest
import kotlin.test.assertContentEquals

class IntCreateTest {
    @Test
    fun createFromIntArray() = runTest {
        val tensor = NDArrayTFJS.int(intArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from int array")

        tensor.close()
    }

    @Test
    fun createFromIntArrayTyped() = runTest {
        val tensor = NDArrayTFJS.int(arrayOf<Int>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from int array")

        tensor.close()
    }

    @Test
    fun createFromByteArray() = runTest {
        val tensor = NDArrayTFJS.int(byteArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from byte array")

        tensor.close()
    }

    @Test
    fun createFromByteArrayTyped() = runTest {
        val tensor = NDArrayTFJS.int(arrayOf<Byte>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from byte array")

        tensor.close()
    }

    @Test
    fun createFromShortArray() = runTest {
        val tensor = NDArrayTFJS.int(shortArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from short array")

        tensor.close()
    }

    @Test
    fun createFromShortArrayTyped() = runTest {
        val tensor = NDArrayTFJS.int(arrayOf<Short>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from short array")

        tensor.close()
    }

    @Test
    fun createFromLongArray() = runTest {
        val tensor = NDArrayTFJS.int(longArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from long array")

        tensor.close()
    }

    @Test
    fun createFromLongArrayTyped() = runTest {
        val tensor = NDArrayTFJS.int(arrayOf<Long>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from long array")

        tensor.close()
    }

    @Test
    fun createFromUIntArray() = runTest {
        val tensor = NDArrayTFJS.int(uintArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from uint array")

        tensor.close()
    }

    @Test
    fun createFromUIntArrayTyped() = runTest {
        val tensor = NDArrayTFJS.int(arrayOf<UInt>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from uint array")

        tensor.close()
    }

    @Test
    fun createFromUByteArray() = runTest {
        val tensor = NDArrayTFJS.int(ubyteArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ubyte array")

        tensor.close()
    }

    @Test
    fun createFromUByteArrayTyped() = runTest {
        val tensor = NDArrayTFJS.int(arrayOf<UByte>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ubyte array")

        tensor.close()
    }

    @Test
    fun createFromUShortArray() = runTest {
        val tensor = NDArrayTFJS.int(ushortArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ushort array")

        tensor.close()
    }

    @Test
    fun createFromUShortArrayTyped() = runTest {
        val tensor = NDArrayTFJS.int(arrayOf<UShort>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ushort array")

        tensor.close()
    }

    @Test
    fun createFromULongArray() = runTest {
        val tensor = NDArrayTFJS.int(ulongArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ulong array")

        tensor.close()
    }

    @Test
    fun createFromULongArrayTyped() = runTest {
        val tensor = NDArrayTFJS.int(arrayOf<ULong>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ulong array")

        tensor.close()
    }
}
