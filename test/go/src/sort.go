package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"sort"
	"strconv"
)

type line struct {
	id 	uint64
	idstr	string
	str 	string
}

type Lines []line
func (a Lines) Len() int           { return len(a) }
func (a Lines) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a Lines) Less(i, j int) bool { return a[i].id < a[j].id }

func main() {
	file, err := os.Open("/var/log/quark/quark.log")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	num := 0
	substr := ""
	lines := []line{}
	for scanner.Scan() {
		num += 1
		fullstr := scanner.Text()
		if len(fullstr) < 13 {
			continue
		}

		str := fullstr[13:]
		if strings.HasPrefix(str, "[ INFO] [") {
			substr = strings.TrimPrefix(str, "[ INFO] [")
		} else if strings.HasPrefix(str, "[ERROR] [") {
			substr = strings.TrimPrefix(str, "[ERROR] [")
		} else {
			continue
		}

		left := strings.Index(substr, "|")
		right := strings.Index(substr, "]")

		if left == -1 || right == -1 {
			continue
		}

		idstr := substr[left+1:right]

		i, _ := strconv.ParseUint(idstr, 10, 64)
		lines = append(lines, line {
			id: i,
			idstr: idstr,
			str: str,
		})
	}

	sort.Sort(Lines(lines))
	lastVal := uint64(0)
	maxVal := uint64(0)
	row := "";
	for _, line := range lines {
		val := line.id
		if lastVal != 0 {
			delta := val - lastVal;
			if delta > maxVal {
				maxVal = delta;
				row = line.str;
			}

		}
		lastVal = val
		fmt.Println(line.str)
	}

	fmt.Printf("MaxVal = %v slowest line is [%v]\n", maxVal, row)
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}
