package io.kinference.operators.operations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class TileTest {
    private fun getTargetPath(dirName: String) = "tile/$dirName/"

    @Test
    fun test_tile() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_tile"))
    }

    @Test
    fun test_tile_precomputed() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_tile_precomputed"))
    }
}
