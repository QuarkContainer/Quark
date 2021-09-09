# Test Cralwer
This is a simple java program to test Quark container environment using webmagic crawler framework
## Build
to build the artifact, use maven commands, the jar file will be under ./target directory
```
cd /path/to/testcrawler
mvn clean package
```
## Run
This project use java8, to run the program, use `-jar` flag
```
java -jar /path/to/target/testcrawler-version-jar-with-dependecies.jar
```
## Known limitation
Currently each crawling thread paused a little after each request with `Thread.sleep()` to avoid getting banned by github.com