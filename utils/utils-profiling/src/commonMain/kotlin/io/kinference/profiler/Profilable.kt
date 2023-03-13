package io.kinference.profiler

interface Profilable {
    fun addProfilingContext(name: String): ProfilingContext
    fun analyzeProfilingResults(): ProfileAnalysisEntry
    fun resetProfiles()
}
