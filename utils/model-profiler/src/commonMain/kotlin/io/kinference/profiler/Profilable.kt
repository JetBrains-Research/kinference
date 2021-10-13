package io.kinference.profiler

import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
interface Profilable {
    fun addContext(name: String): ProfilingContext
    fun analyzeProfilingResults(): ProfileAnalysisEntry
    fun resetProfiles()
}
