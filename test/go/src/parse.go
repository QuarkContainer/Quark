package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"sort"
)

type line struct {
	num int
	str string
}

type Lines []line
func (a Lines) Len() int           { return len(a) }
func (a Lines) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a Lines) Less(i, j int) bool { return a[i].num < a[j].num }

func main() {
	file, err := os.Open("/var/log/quark/quark.log")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	m := make(map[string]line)
	scanner := bufio.NewScanner(file)
	num := 0
	for scanner.Scan() {
		num += 1
		fullstr := scanner.Text()
		if len(fullstr) < 13 {
			continue
		}

		str := fullstr[13:]
		if !strings.HasPrefix(str, "[ INFO] [") {
			continue
		}

		substr := strings.TrimPrefix(str, "[ INFO] [");
		first := strings.Index(substr, "]")
		left := strings.Index(substr, "(")
		right := strings.Index(substr, ")")
		//fmt.Printf("substr: %v, len is %v, i is %v, next is %v \n", substr, len(substr), i, next);
		if len(substr) <= 12 || first==-1 || left == -1 || right == -1 || left > right || left - first != 2 {
			continue;
		}

		idstr := substr[left:right];
		if strings.Index(idstr, "/") == -1 {
			continue;
		}
		//fmt.Printf("idstr is %v\n", idstr);

		m[idstr]= line {
			num: num,
			str: str,
		}
	}

	primes := []line{}
	for _, line := range m {
		primes = append(primes, line)
	}

	sort.Sort(Lines(primes))
	for _, str := range primes {
		fmt.Println(str)
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}
