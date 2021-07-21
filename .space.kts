job("KInference / Build") {
    container("openjdk:11") {
        shellScript {
            content = """
              ./gradlew build  
          """
        }
    }
}

job("KInference / JVM / Test") {
    container("openjdk:11") {
        shellScript {
            content = """
              ./gradlew jvmTest  
          """
        }
    }
}

job("KInference / JVM / Heavy Test") {
    container("openjdk:11") {
        shellScript {
            content = """
              ./gradlew jvmHeavyTest  
          """
        }
    }
}

job("KInference / Release") {
    startOn {
        gitPush {
            enabled = false
        }
    }

    container("openjdk:11") {
        shellScript {
            content = """
              ./gradlew publish    
          """
        }
    }
}
