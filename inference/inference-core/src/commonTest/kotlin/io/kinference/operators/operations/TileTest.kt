package io.kinference.operators.operations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class TileTest {
    private fun getTargetPath(dirName: String) = "tile/$dirName/"

    @Test
    fun test_tile() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_tile"))
    }

    @Test
    fun test_tile_precomputed() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_tile_precomputed"))
    }
}
