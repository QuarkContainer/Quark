#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/timerfd.h>

static char *itimerspec_dump(struct itimerspec *ts);

int main()
{
	int tfd, epfd, ret;
	struct epoll_event ev;
	struct itimerspec ts;
	int msec = 10; // timer fires after 10msec
	uint64_t res;

	printf("testcase start\n");

	tfd = timerfd_create(CLOCK_MONOTONIC, 0);
	if (tfd == -1) {
		printf("timerfd_create() failed: errno=%d\n", errno);
		return EXIT_FAILURE;
	}
	printf("created timerfd %d\n", tfd);

	ts.it_interval.tv_sec = 0;
	ts.it_interval.tv_nsec = 0;
	ts.it_value.tv_sec = msec / 1000;
	ts.it_value.tv_nsec = (msec % 1000) * 1000000;

	if (timerfd_settime(tfd, 0, &ts, NULL) < 0) {
		printf("timerfd_settime() failed: errno=%d\n", errno);
		close(tfd);
		return EXIT_FAILURE;
	}
	printf("set timerfd time=%s\n", itimerspec_dump(&ts));

	epfd = epoll_create(1);
	if (epfd == -1) {
		printf("epoll_create() failed: errno=%d\n", errno);
		close(tfd);
		return EXIT_FAILURE;
	}
	printf("created epollfd %d\n", epfd);

	ev.events = EPOLLIN;
	if (epoll_ctl(epfd, EPOLL_CTL_ADD, tfd, &ev) == -1) {
		printf("epoll_ctl(ADD) failed: errno=%d\n", errno);
		close(epfd);
		close(tfd);
		return EXIT_FAILURE;
	}
	printf("added timerfd to epoll set\n");

	sleep(1);

	memset(&ev, 0, sizeof(ev));
	ret = epoll_wait(epfd, &ev, 1, 500); // wait up to 500ms for timer
	if (ret < 0) {
		printf("epoll_wait() failed: errno=%d\n", errno);
		close(epfd);
		close(tfd);
		return EXIT_FAILURE;
	}
	printf("waited on epoll, ret=%d\n", ret);

	ret = read(tfd, &res, sizeof(res));
	printf("read() returned %d, res=%" PRIu64 "\n", ret, res);

	if (close(epfd) == -1) {
		printf("failed to close epollfd: errno=%d\n", errno);
		return EXIT_FAILURE;
	}

	if (close(tfd) == -1) {
		printf("failed to close timerfd: errno=%d\n", errno);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

static char *
itimerspec_dump(struct itimerspec *ts)
{
    static char buf[1024];

    snprintf(buf, sizeof(buf),
            "itimer: [ interval=%lu s %lu ns, next expire=%lu s %lu ns ]",
            ts->it_interval.tv_sec,
            ts->it_interval.tv_nsec,
            ts->it_value.tv_sec,
            ts->it_value.tv_nsec
           );

    return (buf);
}