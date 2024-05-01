import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.ParameterDisplay.HIDDEN
import jetbrains.buildServer.configs.kotlin.buildFeatures.swabra
import jetbrains.buildServer.configs.kotlin.buildSteps.gradle
import jetbrains.buildServer.configs.kotlin.vcs.GitVcsRoot

version = "2024.03"

project {
    val container = "registry.jetbrains.team/p/ki/containers-ci/ci-corretto-17-firefox:1.0.1"

    val kiVcsRoot = DslContext.settingsRoot
    val kiPrimitivesVcsRoot = GitVcsRoot {
        name = "KInference Primitives"
        id = RelativeId(name.toId())
        url = "https://github.com/JetBrains-Research/kinference-primitives.git"
        branch = "master"
    }

    fun BuildSteps.release(gradleParams: String = "") {
        gradle {
            this.name = "Release"
            tasks = "publish"
            this.gradleParams = gradleParams
            dockerImage = container
        }
    }

    fun ParametrizedWithType.spacePackagesAccessRW() {
        // https://jetbrains.team/extensions/installedApplications/KInference%20Publisher-46Fkb34YGV6Z/authentication
        password("env.JB_SPACE_CLIENT_ID", "credentialsJSON:8622b3eb-ba4f-49aa-ad6c-5aa8b4bc0a3a", display = HIDDEN)
        password("env.JB_SPACE_CLIENT_SECRET", "credentialsJSON:24881bbf-10db-4f8f-943a-05310a0ba714", display = HIDDEN)
    }

    fun BuildFeatures.defaultSwabra() {
        swabra {
            verbose = true
            paths = ""
        }
    }

    vcsRoot(kiPrimitivesVcsRoot)

    buildType {
        name = "Release"
        id = RelativeId(name.toId())
        vcs { root(kiVcsRoot) }
        steps { release("--parallel --console=plain --no-daemon") }
        params { spacePackagesAccessRW() }
        features { defaultSwabra() }
    }

    buildType {
        name = "Release Primitives"
        id = RelativeId(name.toId())
        vcs { root(kiPrimitivesVcsRoot) }
        steps { release() }
        params { spacePackagesAccessRW() }
        features { defaultSwabra() }
    }
}
