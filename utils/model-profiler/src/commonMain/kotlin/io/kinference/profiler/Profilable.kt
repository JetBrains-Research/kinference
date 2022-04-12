package io.kinference.profiler

import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
interface Profilable {
    fun addProfilingContext(name: String): ProfilingContext
    fun analyzeProfilingResults(): ProfileAnalysisEntry
    fun resetProfiles()
}
