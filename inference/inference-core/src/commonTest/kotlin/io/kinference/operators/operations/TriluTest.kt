package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class TriluTest {
    private fun getTargetPath(dirName: String) = "trilu/$dirName/"

    @Test
    fun test_tril() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tril"))
    }

    @Test
    fun test_tril_neg() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tril_neg"))
    }

    @Test
    fun test_tril_one_row_neg()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tril_one_row_neg"))
    }

    @Test
    fun test_tril_out_neg()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tril_out_neg"))
    }

    @Test
    fun test_tril_out_pos()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tril_out_pos"))
    }

    @Test
    fun test_tril_pos()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tril_pos"))
    }

    @Test
    fun test_tril_square()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tril_square"))
    }

    @Test
    fun test_tril_square_neg()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tril_square_neg"))
    }

    @Test
    fun test_tril_zero()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tril_zero"))
    }

    @Test
    fun test_triu() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_triu"))
    }

    @Test
    fun test_triu_neg() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_triu_neg"))
    }

    @Test
    fun test_triu_one_row()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_triu_one_row"))
    }

    @Test
    fun test_triu_out_neg_out()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_triu_out_neg_out"))
    }

    @Test
    fun test_triu_out_pos()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_triu_out_pos"))
    }

    @Test
    fun test_triu_pos()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_triu_pos"))
    }

    @Test
    fun test_triu_square()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_triu_square"))
    }

    @Test
    fun test_triu_square_neg()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_triu_square_neg"))
    }

    @Test
    fun test_triu_zero()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_triu_zero"))
    }
}
