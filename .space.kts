/*job("KInference / Build and Test") {
    host("Build and test") {
        env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
        env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")

        shellScript("Install Firefox, xvfb and JDK") {
            content = """
                apt-get update && apt-get install firefox xvfb openjdk-17-jdk -y -f
            """.trimIndent()
        }

        kotlinScript("Build with Gradle") { api ->
            api.gradlew("build", "-Pci", "-Pdisable-tests", "--console=plain")
        }

        shellScript("Run tests") {
            content = """
                xvfb-run --auto-servernum ./gradlew -Pci jvmTest jsLegacyTest jsIrTest jsTest --console=plain
            """.trimIndent()
        }

        shellScript("Run heavy tests") {
            content = """
                xvfb-run --auto-servernum ./gradlew -Pci jvmHeavyTest jsLegacyHeavyTest jsIrHeavyTest --console=plain
            """.trimIndent()
        }
    }
}*/

job("KInference / Build and Test") {
    container("Build with Gradle", "amazoncorretto:17") {
        addAwsKeys()
//        shellScript {
//            content = """
//                shopt -s extglob
//                test="`find !(serialization) -type d -name 'commonMain'` serialization"
//                echo ${'$'}test
//                for folder in ${'$'}test
//                do
//                mkdir -p ${'$'}JB_SPACE_FILE_SHARE_PATH/${'$'}folder
//                cp -R ${'$'}folder ${'$'}JB_SPACE_FILE_SHARE_PATH/${'$'}folder
//                done
//            """.trimIndent()
//        }

        shellScript {
            content = """
                ./gradlew assemble --parallel --console=plain --no-daemon
                shopt -s extglob
                build_folders="`find !(build) -type d -name 'build'` build"
                for folder in ${'$'}build_folders
                do
                mkdir -p ${'$'}JB_SPACE_FILE_SHARE_PATH/${'$'}folder
                cp -R ${'$'}folder ${'$'}JB_SPACE_FILE_SHARE_PATH/${'$'}folder
                done
            """.trimIndent()
        }
    }

    container("amazoncorretto:17") {
        shellScript {
            content = """
                ls ${'$'}JB_SPACE_FILE_SHARE_PATH
            """.trimIndent()
        }
    }
}

fun Container.addAwsKeys() {
    env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
    env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")
}

/*val packBuildFolders = """
    shopt -s extglob
    build_folders="`find !(build) -type d -name 'build'` build"
    for folder in ${'$'}build_folders
    do
        mkdir -p ${'$'}JB_SPACE_FILE_SHARE_PATH/${'$'}folder
        cp -R ${'$'}folder ${'$'}JB_SPACE_FILE_SHARE_PATH/${'$'}folder
    done
""".trimIndent()*/

job("KInference / Release") {
    startOn {
        gitPush {
            enabled = false
        }
    }

    container("amazoncorretto:17") {
        env["AWS_ACCESS_KEY"] = Secrets("aws_access_key")
        env["AWS_SECRET_KEY"] = Secrets("aws_secret_key")


        shellScript("Release") {
            content = """
                ./gradlew publish
            """.trimIndent()
        }
    }
}
