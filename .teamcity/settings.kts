import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.ParameterDisplay.HIDDEN
import jetbrains.buildServer.configs.kotlin.buildFeatures.provideAwsCredentials
import jetbrains.buildServer.configs.kotlin.buildFeatures.swabra
import jetbrains.buildServer.configs.kotlin.buildSteps.gradle
import jetbrains.buildServer.configs.kotlin.projectFeatures.awsConnection
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
        password("env.JB_SPACE_CLIENT_ID", "", display = HIDDEN)
        password("env.JB_SPACE_CLIENT_SECRET", "", display = HIDDEN)
    }

    fun ParametrizedWithType.publisher() {
        password("env.PUBLISHER_ID", "", display = HIDDEN)
        password("env.PUBLISHER_KEY", "", display = HIDDEN)
    }

    fun BuildFeatures.defaultSwabra() {
        swabra {
            verbose = true
            paths = ""
        }
    }

    vcsRoot(kiPrimitivesVcsRoot)

    features {
        awsConnection {
            id = "AWS_Grazie_Dev_KInference"
            name = "Amazon Web Services (AWS)"
            regionName = "eu-west-1"
            credentialsType = static {
                accessKeyId = "AKIA453I4BURLGEJ5UNX"
                secretAccessKey = "credentialsJSON:7b5b7eb6-e9f0-4788-ad3f-ad0f0a1ead7a"
                stsEndpoint = "https://sts.eu-west-1.amazonaws.com"
            }
            allowInSubProjects = true
            allowInBuilds = true
        }
    }

    buildType {
        name = "Release"
        id = RelativeId(name.toId())
        vcs { root(kiVcsRoot) }
        steps { release("--parallel --console=plain --no-daemon") }
        params {
            spacePackagesAccessRW()
        }
        features {
            provideAwsCredentials { this.awsConnectionId = "AWS_Grazie_Dev_KInference" }
            defaultSwabra()
        }
    }

    buildType {
        name = "Release Primitives"
        id = RelativeId(name.toId())
        vcs { root(kiPrimitivesVcsRoot) }
        steps { release() }
        params {
            publisher()
            spacePackagesAccessRW()
        }
        features {
            provideAwsCredentials { this.awsConnectionId = "AWS_Grazie_Dev_KInference" }
            defaultSwabra()
        }
    }
}
